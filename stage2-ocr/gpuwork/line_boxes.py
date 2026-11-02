#!/usr/bin/env python3
"""Turn the GPU rec daemon's (text, confidence) list into line-box records.

The det stage already produces one quad per text line in FULL-SCAN pixel
coordinates (`gpu_det.py` / `det_boxes()` add the tile y offset), and the rec
stage crops the lines in exactly that order. Padding slots (the batch is
padded to BATCH) have no box, so they are dropped here. The emitted record is
schema-compatible with the CPU worker's line-level output:

    {"text": "...", "confidence": 0.99,
     "bbox": [[x0, y0], [x1, y1]]}       # full-scan pixels, axis-aligned

The detector can place a quad a few px outside the page (it sees the padded
tile), so pass `size=(w, h)` to clamp. Records that clamp to zero area are
dropped. No GPU compute changes -- this is a write-out change only.
"""


def bbox_from_box(box, size=None):
    """Axis-aligned [[x0,y0],[x1,y1]] ints from a det quad (N x 2 points)."""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    if size is not None:
        w, h = int(size[0]), int(size[1])
        x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
        y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
    return [[x0, y0], [x1, y1]]


def emit_texts(texts, boxes, keep_empty=False, size=None):
    """Zip CTC results with their det boxes; drop padding and empty lines."""
    out = []
    for i, (text, conf) in enumerate(texts):
        if i >= len(boxes):
            break                      # padding slot, no det box
        text = str(text)
        if not keep_empty and not text.strip():
            continue
        bbox = bbox_from_box(boxes[i], size)
        if bbox[1][0] <= bbox[0][0] or bbox[1][1] <= bbox[0][1]:
            continue                   # clamped to nothing: off-page line
        out.append({"text": text,
                    "confidence": round(float(conf), 4),
                    "bbox": bbox})
    return out


if __name__ == "__main__":
    # Tiny self-check, no GPU needed: python3 line_boxes.py
    boxes = [[[10, 20], [110, 20], [110, 60], [10, 60]],
             [[300, 500], [400, 500], [400, 560], [300, 560]]]
    texts = [("hello", 0.9), ("", 0.0), ("", 0.0), ("", 0.0)]
    assert emit_texts(texts, boxes) == [
        {"text": "hello", "confidence": 0.9,
         "bbox": [[10, 20], [110, 60]]}], emit_texts(texts, boxes)
    assert emit_texts([], []) == []
    # clamping + degenerate drop
    edge = [[[-4, 5], [50, 5], [50, 9], [-4, 9]],
            [[10, 5], [10, 9], [10, 9], [10, 5]]]
    assert emit_texts([("edge", .5), ("zero", .9)], edge, size=(40, 40)) == [
        {"text": "edge", "confidence": 0.5, "bbox": [[0, 5], [39, 9]]}], \
        emit_texts([("edge", .5), ("zero", .9)], edge, size=(40, 40))
    print("line_boxes.py self-check OK")
