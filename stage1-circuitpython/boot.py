# SPDX-FileCopyrightText: 2022 Neradoc
# SPDX-License-Identifier: MIT
"""
boot.py

Enables a USB keyboard plus a custom *absolute* mouse HID device.
The absolute mouse reports coordinates in the range 0..32767 instead
of relative pixel deltas.

This file only runs at boot, so you must reset the board after changing
it for the new USB devices to appear.

The absolute mouse has the same Usage Page (0x01) and Usage (0x02) as a
normal (relative) mouse. To keep adafruit_hid's `find_device()` from
picking the wrong one, we only enable the keyboard and the absolute
mouse here (no regular MOUSE device).
"""
import usb_hid
from absolute_mouse.descriptor import device

# usb_hid.devices[0] -> keyboard
# usb_hid.devices[1] -> absolute mouse
usb_hid.enable((usb_hid.Device.KEYBOARD, device))