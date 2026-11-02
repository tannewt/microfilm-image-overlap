# Remote computers via incus (dalinar, clanker, majormajor)

How to reach and use the other machines from hoid. hoid runs the incus
**client only** — there is no local daemon (`/var/lib/incus/unix.socket` does
not exist, `systemctl` is not available in this sandbox). Every instance
lives on a remote host, reached over Tailscale at `https://<100.x>:8443` with
TLS client certificates.

Facts below were verified 2026-10-28 from hoid, and cross-checked against the
Oct 15-21 pi sessions that built and ran the OCR fleet. State changes: always
list before assuming.

Related docs: `SCAN_OCR_RUN.md` (the 9,818-scan distributed run),
`GLM-OCR.md` (gpu1 / GLM / PP GPU workers), `WORKERS.md` + `DISTRIBUTE.md`
(the HTTP worker-as-a-service pattern, independent of incus).

---

## 1. Remotes, credentials, project

Remotes are already configured in `~/.config/incus/`:

```
~/.config/incus/config.yml      # remotes + default-remote: local
~/.config/incus/client.crt/.key # our client cert (trusted by the hosts)
~/.config/incus/servercerts/    # pinned server certs
```

```bash
incus remote list
# clanker     https://100.82.76.119:8443    incus  tls
# dalinar     https://100.112.157.138:8443  incus  tls
# majormajor  https://100.110.170.53:8443   incus  tls
# images      https://images.linuxcontainers.org  simplestreams
# local       unix://                       <- no daemon on hoid; unusable
```

Two rules cover most confusion:

1. **Always name the remote** (`dalinar:gpu1`, `clanker:ocr1`). Without a
   remote you hit `local:` and get "incus daemon doesn't appear to be
   started".
2. **Always pass `--project sandboxes`.** All the OCR machines live in the
   `sandboxes` project; the `default` project is empty, so a bare
   `incus list dalinar:` shows nothing.

```bash
incus list dalinar:  --project sandboxes -c ns4   # name,state,ipv4,type
incus list clanker:  --project sandboxes -c ns4
incus list majormajor: --project sandboxes        # currently times out
incus list dalinar: --all-projects                # includes default + sandboxes
```

`--project` may appear before or after the instance target; the examples here
put it after, which is what the fleet sessions used.

If a remote stops answering: `ping -c1 <100.x>` first, then
`incus info <remote>:`. `100.x` is Tailscale — hoid's own address is
`100.104.156.160` (`ip -brief addr show tailscale0`). The remotes' host
firewall is not involved; the incus daemon listens on `:8443`.

---

## 2. Hosts

| Remote | Tailscale | Host hardware | Host OS / kernel | Incus | GPU |
|---|---|---|---|---|---|
| `dalinar` | 100.112.157.138 | 32c, 70.9 GB | Arch, 7.2.11-arch1-1 | 7.0.0 (lxc) | RX 9070 XT 16 GB (`0000:0d:00.0`) |
| `clanker` | 100.82.76.119 | 32c, 137 GB (Ryzen AI MAX+ 395) | Arch, 7.2.9-arch1-1 | 7.0.0 (lxc+qemu) | Radeon 8060S iGPU (Strix Halo) |
| `majormajor` | 100.110.170.53 | 12c, 34.4 GB (last seen) | — | — | RX 5500M (last seen; was due a 9070 XT) |
| `hoid` (here) | 100.104.156.160 | 16c, 92 GB | container, no incus daemon | client only | none usable |

`majormajor` was **offline on 2026-10-28** (ping and `:8443` both time out).
Earlier sessions used `majormajor:ocr1` successfully; treat it as an
optional fourth box that may be asleep, and check before planning around it.

Refresh hardware with:

```bash
incus query dalinar:/1.0/resources | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("cpu", d["cpu"]["total"], "mem GB", round(d["memory"]["total"]/1e9,1))
print("gpus", [(g.get("vendor"),g.get("product")) for g in d["gpu"]["cards"]])'
incus info dalinar: | grep -E 'driver|kernel|version'   # daemon + host
```

---

## 3. Instances (project `sandboxes`)

Limits and contents are the important part; container IPs are DHCP on each
host's `incusbr0` and **change on restart** (clanker `10.179.28.0/24`,
dalinar `10.11.108.0/24`).

| Instance | OS | Limits | Root disk | Role / what is installed |
|---|---|---|---|---|
| `dalinar:gpu1` | Arch | 20c / 36 GiB | 48 GiB + `ocrdata` volume | PP GPU + GLM box. `gpu` + `/dev/kfd` devices, `/mnt/ocrdata` (331 GB btrfs volume: models, ollama, venv-glm, `glm-batch/`), `/mnt/sandbox` (host `/home/tannewt/sandbox`), `/opt/rocm`, `/opt/mgx16`, `/root/gpuwork` (migraphx det/rec daemons), `/root/models`, `/root/onnx_models`; `/root/ocrwork` now also has a `.venv-paddle` CPU worker env (added 2026-10-28). `gpu-ocr@det` + `gpu-ocr@rec` enabled (they start with the container); `ollama-glm` installed but **disabled** (start manually; it and PP share the 16 GB VRAM) |
| `dalinar:ocr1` | Ubuntu 24.04 | 12c / 12 GiB | 15 GiB | CPU OCR worker. `/root/ocrwork` with `.venv-paddle` (paddle 3.2.1, paddleocr 3.7.0), PP-OCRv6 det+rec models cached in `/root/.paddlex/official_models`, `ocr_puller.py`, `setup_paddle_env.sh`, `requirements-ocr.txt`, puller logs. DNS fixed via `/etc/systemd/resolved.conf.d/dns.conf` (`DNS=1.1.1.1 8.8.8.8`) |
| `clanker:ocr1` | Ubuntu 24.04 | 12c / 14 GiB | 25 GiB | CPU OCR worker + GPU experiments. `gpu0` + `/dev/kfd` attached (iGPU), `.venv-paddle`, onnx/ncnn workers, `onnx_models/`, llama-server models, det/parity test scripts, puller logs |
| `clanker:ocr2` | Ubuntu 24.04 | 12c / 14 GiB | 25 GiB | CPU OCR worker. `.venv-paddle`, worker + puller scripts, logs |
| `clanker:ocr-arch` | Arch | 6c / 64 GiB | 70 GiB | iGPU MIGraphX box. `gpu0` + `/dev/kfd`, `/opt/rocm`, `/opt/mgx16`, `/root/models`, `/root/gpuwork` (same daemon scripts), no paddle venv |
| `clanker:warm-zephyr` | Ubuntu 24.04 | 2c / 4 GiB | default | Small always-on canary/scratch box (was used to check DNS/network) |
| `majormajor:ocr1` | Ubuntu 24.04 | 10c / 13 GiB (historical) | 25 GiB | Was staged as a CPU/GPU worker; remote offline |

Project-wide caps (instances' limits are per instance):

```bash
incus project show dalinar:sandboxes | grep -E 'limits|restricted'
#   limits.cpu: 32     limits.memory: 48GiB    restricted: true
incus project show clanker:sandboxes | grep -E 'limits|restricted'
#   limits.cpu: 32     limits.memory: 96GiB
```

`restricted: true` permits only `/home/tannewt/sandbox` for disk devices plus
`gpu` and `unix-char`. **Proxy devices are not allowed**, which is why the
fleet is pull-based: containers reach out to hoid (`100.104.156.160:8076`) and
hoid cannot dial into a container. Nesting is blocked (`restricted.containers.nesting: block`).

---

## 4. Command cookbook

### Inspect

```bash
incus list dalinar: --project sandboxes -c nst4       # state + addresses
incus config show dalinar:gpu1 --project sandboxes    # limits, devices, mounts
incus info --show-log dalinar:gpu1 --project sandboxes # last boot log if stuck
incus exec dalinar:gpu1 --project sandboxes -- uname -a
incus exec clanker:ocr1 --project sandboxes -- bash -lc 'df -h /; free -g'
```

`incus exec` needs `--` between the instance and the command. Non-interactive
by default; use `bash -lc '...'` for pipes/loops.

### Start / stop (containers are stopped by default)

```bash
incus start dalinar:gpu1 --project sandboxes
incus stop  dalinar:gpu1 --project sandboxes
incus restart clanker:ocr1 --project sandboxes
```

Stopping a container destroys all **transient** `systemd-run` units inside
it; long jobs launched that way do **not** come back. Anything that should
survive a restart needs a real unit file or a host-side `incus exec` relaunch.

### Copy files (works on stopped containers too)

```bash
incus file push local.py dalinar:gpu1/root/gpuwork/local.py --project sandboxes
incus file push -r gpuwork/ dalinar:gpu1/root/gpuwork/     --project sandboxes
incus file pull dalinar:gpu1/root/gpuwork/gpu_rec.py ./gpu_rec.py --project sandboxes
incus exec clanker:ocr1 --project sandboxes -- chmod +x /root/ocrwork/run.sh
```

### Launch a new scratch worker

```bash
incus launch images:ubuntu/24.04 clanker:ocr3 --project sandboxes \
  -c limits.cpu=8 -c limits.memory=12GiB -d root,size=20GiB
incus exec clanker:ocr3 --project sandboxes -- bash -lc 'apt-get update'
incus delete clanker:ocr3 --project sandboxes --force
```

Use `images:ubuntu/24.04` (public simplestreams) or an image already cached
on that host (`incus image list dalinar:`). `-d root,size=` sets the container
root disk; big data goes on a host-backed disk (`/mnt/sandbox`, or the
`ocrdata` volume on gpu1) rather than a bigger root.

### Live resource changes

```bash
incus config set clanker:ocr2 limits.memory=6GiB --project sandboxes
incus project set clanker:sandboxes limits.memory=96GiB
incus config device add dalinar:gpu1 w8077 proxy \
  listen=tcp:0.0.0.0:8077 connect=tcp:127.0.0.1:8077 --project sandboxes # DENIED
```

The proxy add fails by design (`restricted.devices.proxy` not allowed), so
plan for pull/exec-based access instead of exposing ports.

### Long-running work inside a container

The fleet pattern (from `SCAN_OCR_RUN.md`), survives a dropped SSH/incus
connection, not a container restart:

```bash
incus exec clanker:ocr1 --project sandboxes -- systemd-run \
  --unit=ocrpull1 --property=Restart=always --property=RestartSec=30 \
  --property=Environment=OMP_NUM_THREADS=4 \
  --property=Environment=OCR_HOST=clanker:ocr1 \
  bash -c 'cd /root/ocrwork && exec .venv-paddle/bin/python ocr_puller.py \
           http://100.104.156.160:8076 clanker:ocr1 >> puller1.log 2>&1'
incus exec clanker:ocr1 --project sandboxes -- systemctl status ocrpull1 --no-pager
incus exec clanker:ocr1 --project sandboxes -- systemctl stop ocrpull1
```

For a one-off that only needs to outlive the current shell but not the
container, `setsid nohup ... &` works the same as on a normal host.

---

## 5. Network model

```
hoid (100.104.156.160, tailscale)
  │  pull: GET /job, POST /result (HTTP)
  ▼
clanker incusbr0 10.179.28.0/24 ── ocr1 .54, ocr2 .191, ocr-arch .194, warm-zephyr .192
dalinar incusbr0 10.11.108.0/24 ── gpu1, ocr1
  └─ containers have NAT internet; DNS must be fixed (below)
```

- Container → hoid over Tailscale works (that is how every puller fetched
  jobs from `http://100.104.156.160:8076`).
- hoid → container is **exec/file only**. There is no inbound port, so test
  worker HTTP servers from inside the container (`curl localhost:8077/health`
  via `incus exec`), not from hoid.
- IPs are per-host subnets and change on restart; do not hard-code them.

### DNS inside Ubuntu containers (needed once per container)

The incus dnsmasq gateway REFUSES recursion, and `127.0.0.53` has no working
upstream, so `apt`/`uv`/HuggingFace hang or fail. Fix (already applied on
`dalinar:ocr1`, `clanker:ocr1`, `clanker:ocr2`):

```bash
incus exec clanker:ocr1 --project sandboxes -- bash -c '
  mkdir -p /etc/systemd/resolved.conf.d
  printf "[Resolve]\nDNS=1.1.1.1 8.8.8.8\n" > /etc/systemd/resolved.conf.d/dns.conf
  systemctl restart systemd-resolved'
incus exec clanker:ocr1 --project sandboxes -- getent hosts astral.sh
```

The Arch container (`ocr-arch`) needed `/etc/hosts` pins for uv/HF because
UDP/53 was blocked at the time.

---

## 6. What runs where (OCR)

| Job | Where | How |
|---|---|---|
| PP-OCRv6 CPU (scan-level) | `dalinar:ocr1`, `clanker:ocr1`, `clanker:ocr2`, plus 3 pullers on `dalinar:gpu1` (2026-10-28) | `ocr_puller.py` + `ocr_worker_server.py`, `.venv-paddle`, `OCR_WORD_BOX=0` |
| PP-OCRv6 GPU (MIGraphX) | `dalinar:gpu1`, `clanker:ocr-arch` | `gpu-ocr@det`/`@rec` (gpu1) or `/root/gpuwork` daemons; see `GLM-OCR.md` |
| GLM-OCR / Ollama | `dalinar:gpu1` only | `ollama-glm` (disabled at boot; stop PP GPU units first — 16 GB VRAM) |
| Master / dispatcher | hoid | `ocr_master_scans.py`, `ocr_master_1006.py`, `dispatch_ocr.py` |
| Scratch/canary | `clanker:warm-zephyr` | 2c/4G, always on |

VRAM is the scarce resource on gpu1: run either the PP GPU workers or
`ollama-glm`, never both.

Current corpus run (2026-10-28/29): the 20261028 batch, master on hoid, CPU
pullers on `dalinar:ocr1` + `clanker:ocr1/ocr2`, GPU pullers on `dalinar:gpu1`
+ `clanker:ocr-arch`. Setup, commands and the engine-consistency fix are in
`SCAN_OCR_RUN.md` ("Run 2026-10-28/29") and `gpuwork/README.md`.

`clanker:ocr-arch` has no `gpu-ocr@` unit; start its daemons with
`systemd-run --unit=gpu-ocr-det/rec --working-directory=/root/gpuwork
--property=Environment=LD_LIBRARY_PATH=/opt/mgx16/lib
--property=Environment=PYTHONPATH=/opt/mgx16/lib:/root/gpuwork
bash -c '/root/.venv-onnx/bin/python gpu_daemon.py det|rec >> det|rec.log 2>&1'`
and then run `ocr_puller_gpu2.py` with `GPU_HOST=clanker:ocr-arch`.

---

## 7. Gotchas learned the hard way

- **Empty `incus list`** = wrong project. Add `--project sandboxes`.
- **"daemon doesn't appear to be started"** = you hit `local:`. Prefix the
  remote or use `--remote`.
- **majormajor timeouts** are normal when it is powered off; ping first.
  Do not queue work that only it can run.
- **Transient units die** on container stop/restart — relaunch or install a
  unit file.
- **OOM kills** from per-container memory caps: a paddle puller is ~2.6 GB
  RSS, so 6 GiB fits two pullers, not four. The project cap (48 GiB dalinar,
  96 GiB clanker) is shared by all instances on the host.
- **Disk**: container roots are 15-70 GiB. Data belongs on
  `/mnt/ocrdata` (gpu1, 331 GB) or `/mnt/sandbox` (host-shared scratch,
  allow-listed for the project); `/mnt/sandbox` is not present on the
  clanker containers.
- **DNS** breaks apt/uv/HF after a rebuild; apply the section-5 fix first.
- **No proxy devices** — cannot expose a container port to hoid. Pull, or
  `incus exec` into the container to test.
- **Cert restrictions**: the client cert is trusted but the project is
  `restricted`; denied operations are usually project policy, not TLS.
- **`incus start gpu1` starts the PP GPU services** (`gpu-ocr@det/rec` are
  enabled). Expect `/dev/shm/gpuw` activity and the GPU to be busy.

---

## 8. Quick reference

```bash
# inventory
incus remote list
incus list dalinar:  --project sandboxes -c ns4
incus list clanker:  --project sandboxes -c ns4

# get on a box
incus start dalinar:gpu1 --project sandboxes
incus exec dalinar:gpu1 --project sandboxes -- bash -l

# move a script and run it
incus file push gpuwork/gpu_daemon.py dalinar:gpu1/root/gpuwork/ --project sandboxes
incus exec dalinar:gpu1 --project sandboxes -- systemctl restart gpu-ocr@rec

# GPU line-box check (see gpuwork/README.md); push a scan to test_scan.tif first
incus exec dalinar:gpu1 --project sandboxes -- bash -lc '
  cd /root/gpuwork
  export LD_LIBRARY_PATH=/opt/mgx16/lib PYTHONPATH=/opt/mgx16/lib:/root/gpuwork
  /root/.venv-onnx/bin/python daemon_client_test.py /root/gpuwork/test_scan.tif
  /root/.venv-onnx/bin/python validate_texts.py /dev/shm/gpuw/texts.json \
      /dev/shm/gpuw/boxes.npz --scan /root/gpuwork/test_scan.tif'

# fleet puller (pull model; master on hoid)
incus exec clanker:ocr1 --project sandboxes -- systemd-run \
  --unit=ocrpullA --property=Restart=always \
  --property=Environment=OMP_NUM_THREADS=4 \
  bash -c 'cd /root/ocrwork && exec .venv-paddle/bin/python ocr_puller.py \
           http://100.104.156.160:8076 clanker:ocr1 >> pullerA.log 2>&1'

# leave things as you found them
incus stop clanker:ocr1 clanker:ocr2 clanker:ocr-arch --project sandboxes
incus stop dalinar:ocr1 --project sandboxes
incus stop dalinar:gpu1 --project sandboxes     # also stops gpu-ocr@*
```
