# Regime D on rented Indian GPUs — complete runbook

Zero prior cloud experience assumed. Every click, every command, in order.

**What you are building:** two cloud GPU machines running stock vLLM — an L4 and an
A30, deliberately different memory bandwidth. The L4 node *also* runs the DIO
gateway and the experiment harness. The harness drives traffic through DIO, DIO
routes it to whichever GPU it predicts will be faster, and we measure whether that
beats round-robin.

**Why the gateway sits on the L4 rather than its own box:** co-locating does give
the local worker a small latency edge, so the choice of host matters. Putting it on
the **slow** card points that edge at the worker the paper claims is worse — the
bias runs against the result, so it cannot manufacture the win. A third neutral
node would be marginally cleaner, but it also has to exist in the same region as
your GPUs, and CPU and GPU stock are allocated separately. §6.2 has the full
argument.

**Cost:** about ₹139/hour for both nodes, roughly ₹820 for a complete campaign
including setup and GST. See §9 before you start.

**Where you type things.** Everything below happens in one of three places, and the
runbook marks each block:

- **Browser** — the E2E console, for §1–§7 (account, KYC, payment, nodes).
- **PowerShell on your laptop** — only to `ssh` in, `scp` files up, and `scp`
  results back. Open one window per node and leave them open.
- **Inside an SSH session** — every `bash ...` command. These run *on the cloud
  machines*, not on your laptop.

You do **not** need VS Code, Docker, or WSL for any of this. Windows 10+ ships
with `ssh` and `scp` built in. If you would rather read logs in a GUI, VS Code's
**Remote-SSH** extension can open a cloud node as a normal editor window — nice to
have, never required.

---

## 0. Before you touch a browser

You need:

- **Aadhaar** linked to a mobile number that can receive OTP (for KYC via DigiLocker)
- **PAN card** number
- **A UPI app** — GPay / PhonePe / Paytm / BHIM all work
- **An SSH key.** Already generated for you at `~/.ssh/dio_cloud`. Your public key
  is in `~/.ssh/dio_cloud.pub` — you will paste its contents into the console later.
  Print it with:

  ```powershell
  Get-Content $env:USERPROFILE\.ssh\dio_cloud.pub
  ```

  It is one line starting `ssh-ed25519 AAAAC3...`. **Only ever paste the `.pub`
  file.** The file without `.pub` is the private key — it must never leave your
  laptop, never be pasted into a website, never be shared.

**Provider: E2E Networks** (`myaccount.e2enetworks.com`). Chosen because it is the
one Indian GPU host that documents UPI payment explicitly, bills hourly with no
minimum commitment, and stocks L4, A30, A40, L40S and A100. No credit card needed.

---

## 1. Create the account

1. Go to **https://myaccount.e2enetworks.com/** → **Sign Up**.
2. Enter name, email, mobile, password. Verify the email link and the mobile OTP.
3. Sign in. You land on the MyAccount dashboard.

## 2. Complete KYC (do this first — you cannot launch GPUs without it)

1. Dashboard → your profile / **Complete KYC** prompt.
2. Choose **Aadhaar / DigiLocker**. You are redirected to DigiLocker.
3. Enter your Aadhaar number → OTP to your Aadhaar-linked mobile → consent to share.
4. Enter your **PAN**.
5. Wait for approval. Usually minutes; can take a few hours. **You cannot skip this
   and you cannot launch a GPU node until it clears**, so start it before you plan
   to run anything.

## 3. Load money (UPI, no card)

E2E is prepaid: you top up "Infra Credits" and machines burn them hourly.

1. Sidebar → **Billing** → **Add Credits** (or **Recharge**).
2. Amount: **₹3,000** for a first campaign. That is comfortable headroom over the
   ~₹1,800 worst case and leaves room for one mistake.
3. Payment method → **PayNow (Razorpay)** → **UPI**.
4. Either scan the QR with your UPI app or enter your UPI ID and approve the
   collect request in the app.
5. Confirm the balance appears in **Billing**. It is usually instant.

> **Fraud warning, straight from E2E's own docs:** if you pay by NEFT/RTGS into a
> virtual account instead of UPI, use *only* the account shown in your live console
> at that moment (currently **RBL Bank**). Bank details have changed historically
> and money sent to an old account is gone. UPI via the console avoids this
> entirely — prefer it.

## 4. Upload your SSH key

1. Sidebar → **Settings** → **SSH Keys** (may appear as **My Account → SSH Keys**).
2. **Add New Key**.
3. **Name:** `dio_cloud` (same as the filename, so the console dropdown and your
   `ssh -i` flag can never drift apart)
4. **Public Key:** paste the *entire single line* from `dio_cloud.pub`, including
   the `ssh-ed25519` prefix and the `dio-regime-d` comment at the end.
5. Save. It now appears as a selectable key when you create nodes.

## 5. Create the security group (do this BEFORE creating nodes)

The nodes must talk to each other on port 8000, and only to each other. A node
created without this rule will look completely healthy while the gateway sees
nothing but connection timeouts — and both GPUs bill the whole time you debug it.

1. Sidebar → **Network** → **Security Groups** → **Create Security Group**.
2. **Security Group Name:** `dio-regime-d` (letters, numbers, `_`, `-` only).
3. **Inbound rule 1** — SSH so you can log in:
   - Protocol: `SSH` (or `Custom TCP` with port `22`)
   - Port Range: `22`
   - Source IPv4 / Network: **My IP / Whitelist IP** (auto-fills your public IP)
4. **Inbound rule 2** — vLLM. The gateway does not exist yet, so what you put as
   the source depends on whether your VPC request came through (§6.1):
   - Protocol: `Custom TCP`
   - Port Range: `8000`
   - Source, **with a VPC**: the VPC's CIDR — the exact range you entered in §6.1,
     e.g. `10.20.0.0/16`. Private by definition, so this is already correct and
     §7 becomes a no-op.
   - Source, **without a VPC**: your own public IP as `/32` for now, then replace
     it with the gateway's public IP in §7 once the node exists.

   Never `0.0.0.0/0` on this one. Port 8000 is an unauthenticated LLM endpoint —
   anyone scanning the range can spend your credit on it.
5. **Outbound rule** — required, at least one must exist:
   - Protocol: `All Traffic` (or `Custom TCP`, ports `1-65535`)
   - Destination: `0.0.0.0/0`

   The nodes need outbound internet to download vLLM and the model weights.
6. **Create Group**.

> Note: **at least one inbound and one outbound rule are mandatory** — E2E rejects
> the group otherwise.

**Two inbound rules is the whole list.** In particular you do *not* need a rule for
the DIO gateway itself: it binds `127.0.0.1` on ports 19300+
(`run_regime_d_hetero.py:228`) and the harness reaches it over loopback on the same
node, so it is never exposed to the network at all. Nothing to open, nothing to
lock down.

**Outbound has to stay open.** Setup pulls ~5–8 GB of torch/CUDA wheels from PyPI
and ~6 GB of weights from HuggingFace. Restricting egress on a node that exists
for five hours breaks the install and buys you nothing.

**If your home IP rotates and SSH stops working:** the run does not die. It lives in
tmux on the gateway, detached from your session. Update inbound rule 1 to your new
IP in the console, reconnect, `tmux attach -t dio`. You cannot lock yourself out
permanently — the security group is always editable from the browser.

## 6. Create the two nodes

Sidebar → **Compute** → **Nodes** → **Create Node**, twice.

For **both** nodes: OS image **Ubuntu 22.04** (pick the CUDA/NVIDIA variant if one
is offered; 24.04 also works — the scripts install into a venv, so its
externally-managed-Python rule is not a problem), SSH key **`dio_cloud`**,
security group **`dio-regime-d`**, same region for both.

> **The image must have NVIDIA drivers.** `setup_worker.sh` stops immediately if
> `nvidia-smi` is missing rather than half-installing vLLM against no GPU. If E2E
> offers a plain Ubuntu and a CUDA/NVIDIA Ubuntu, take the CUDA one.

> **Do not use a T4 as one of the pair.** T4 is 320 GB/s — within 7% of the L4's
> 300 GB/s. Pairing them gives you no bandwidth contrast at all and the experiment
> measures nothing. T4 *as the slow worker against the A30* is fine (320 vs 933,
> a 2.9x contrast) and has a bonus: Regime A already ran on T4s, so the recovered
> T4 slope becomes a cross-regime consistency check. Just never T4 + L4.

| # | Name | Plan | Why |
|---|------|------|-----|
| 1 | `dio-l4` | **L4 24GB** — ₹49/hr | the slower SKU (300 GB/s); **also runs the gateway** |
| 2 | `dio-a30` | **A30 24GB** — ₹90/hr | the faster SKU (933 GB/s) |

There is no separate CPU node: the gateway rides on the L4. See §6.2 for why that
is sound and why the *slow* card is the right host for it.

**Why A30 and not A100.** Decode is memory-bandwidth-bound, so the SKU difference
the gateway has to learn lives in bandwidth, not in VRAM or price. A30 is 933 GB/s
of HBM2 at ₹90/hr; A100-40 is 1555 GB/s at ₹179/hr. Against the L4's 300 GB/s that
is a 3.1x contrast versus 5.2x — both far more than enough to learn, and the A30
costs half as much. Both cards are also 24 GB, so declared VRAM is identical across
workers and cannot be raised as a confound.

Other valid substitutions, by memory bandwidth (the only spec that matters here):

| GPU | Bandwidth | ₹/hr | Note |
|-----|-----------|------|------|
| L4 24GB | 300 GB/s | 49 | the slow worker in every pairing |
| A40 48GB | 696 GB/s | 96 | fine, but pricier than A30 and slower |
| L40S 48GB | 864 GB/s | 102 | fine; GDDR6, slightly under A30 |
| **A30 24GB** | **933 GB/s** | **90** | best bandwidth per rupee here |
| A100 40GB | 1555 GB/s | 179 | biggest contrast, ~2x the cost |

> **Avoid pairing A40 with L40S.** They are 696 vs 864 GB/s — a 1.24x gap at nearly
> identical price. That is not two tiers, it is two near-clones, and a 24% slope
> difference is unlikely to separate from noise at n=10. If you want a *third*
> worker, make it a genuinely distinct tier (L4 + A40 + A100-40 = 300/696/1555)
> and expect ~₹324/hr. The harness and both paper scripts handle N workers, but
> the two-worker pairing is what the paper's tables and ratio row are built for.


**Same region matters.** Cross-region nodes add tens of milliseconds of *variable*
network latency between the gateway and one worker, which contaminates the
measurement. (Stable latency would be harmless — it lands in the per-worker
intercept and cancels out during ranking. Jitter does not.)

### 6.1 VPC: request one, but do not wait on it

E2E asks for a VPC and new accounts often have none available, only a "request"
button. Submit the request — a VPC is free, gives you `10.x.x.x` private IPs, and
lets you keep port 8000 off the public internet entirely. That is the clean path.

**If "E2E provided IPv4 CIDR" says the plan is temporarily unavailable,** switch the
CIDR option to **custom**. That is not a downgrade — E2E's own note says provided
CIDRs cannot be subnetted and custom ones can, so custom is the more capable
choice. Values that work:

| Field | Value |
|-------|-------|
| VPC Name | whatever it prefills (`VPC-936`) is fine |
| CIDR Option | **Custom** |
| IPv4 CIDR | `10.20.0.0/16` |
| Subnet (if asked) | `10.20.1.0/24` |

Any RFC1918 range is valid (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`).
`10.20.0.0/16` is suggested only because it is far from the `10.0.0.x` defaults
that provider-internal services tend to occupy, which removes any chance of a
collision you would have to diagnose later. Three nodes need three addresses; a
`/16` is 65,534 of them, so sizing is not something to think about.

Whatever range you enter becomes two things: the source CIDR for the port-8000
inbound rule in §5, and the range your workers' private IPs come from in §8.5. If
you pick `10.20.0.0/16`, the rule source is `10.20.0.0/16` — tighter and more
correct than the generic `10.0.0.0/8`.

If custom is blocked too, click **Request**, then check whether the node form
actually refuses to submit without a VPC or merely warns. If it truly blocks you,
you are waiting on E2E; if it only warns, proceed without one on the terms below.

**If approval is slow, run without it.** Public IPs work, and the reason is worth
understanding rather than taking on faith: the model is `y ≈ s·tokens + b`, and a
network hop adds a roughly *constant* per-request cost. A constant lands in the
intercept `b`, not the slope `s` — and `s`, plus the ordering of `s` across
workers, is the entire result. So a stable extra millisecond changes nothing that
the paper reports. Both nodes sit in the same datacenter, so the traffic does not
leave the building and jitter stays small.

Running without a VPC, two things become mandatory rather than optional:

- In §8.5, pass the A30's **public** IP instead of its private one. The L4 stays
  `127.0.0.1` either way — it is the same machine as the gateway.
- In §7, the port-8000 rule must be locked to the gateway's **public** IP —
  `/32`, one address, not `0.0.0.0/0`. An open port 8000 is an unauthenticated
  LLM endpoint that anyone scanning the range can bill you for.

Either way you also need a **public IP on at least the gateway** so you can SSH
in. Simplest is a public IP on both; then you can `ssh` to each directly
instead of tunnelling through the other. "Reserve IPv4" means a *static* address
that survives a reboot — you do not need that for a 5-hour run, but do confirm the
node gets some public address, because a VPC-only node with no public IP is one
you cannot log into.

After each node boots, write down from the node list:

- **Public IP** (to SSH in from your laptop)
- **Private IP** (for gateway → worker traffic; usually `10.x.x.x`)

Fill this in — you will use it constantly:

```
dio-l4  (worker + gateway)  public: ______________  private: ______________
dio-a30 (worker)            public: ______________  private: ______________
```

> **If A30 shows as out of stock:** substitute **L40S 48GB** (₹102/hr) or **A40
> 48GB** (₹96/hr) — either still gives a >2x bandwidth contrast against the L4.
> Any SKU pair with genuinely different memory bandwidth works; the experiment is
> about heterogeneity, not about a specific card. Update the `--vram` argument in
> §8.5 to match (both are 48 GB, so `a30=24000` becomes e.g. `l40s=48000`), and
> record the exact GPU string `start_vllm.sh` prints — the paper reports real
> hardware.

### 6.2 No CPU node in your GPU region? Skip the CPU node

Availability is per-region, and CPU and GPU stock do not track each other. If the
region with your GPUs has no CPU plan, **do not rent across regions** — that is the
one configuration to avoid, because inter-region jitter is variable and lands in
the measurement rather than cancelling out.

Rent **two GPU nodes only** and run the gateway on one of them. This needs no code
changes: `run_all.sh` takes the worker URLs as arguments and builds its own CPU-only
venv (`~/dio-gw-venv`) separate from the worker's `~/dio-venv`, and the gateway is a
FastAPI reverse proxy that binds loopback — it is not a second service competing for
the GPU. A 12-vCPU node runs a throttled sequential load generator and a vLLM API
server without either noticing the other.

**Put the gateway on the *slower* card.** That worker then answers over loopback
while the fast worker takes a network hop, so the small constant advantage goes to
the card the paper claims is worse. The bias runs *against* your own result, which
means no reviewer can attribute the win to a network artefact. Doing it the other
way round would be the version you have to defend.

Then §8 changes in exactly two places:

- **§8.4:** run the gateway setup on the slow GPU node instead of `dio-gw`. It
  already has git and python3 from `setup_worker.sh`.
- **§8.5:** the local worker's URL becomes `http://127.0.0.1:8000`:

```bash
bash ~/Go-serve/dio-serve/scripts/cloud/run_all.sh \
  "l4=http://127.0.0.1:8000,a30=http://<A30_PRIVATE_IP>:8000" \
  "l4=24000,a30=24000"
```

Saves ~₹15 and one machine's worth of setup. Record in the paper that the gateway
was co-located with the L4 worker — it is a detail a careful reader will want, and
it reads as rigour rather than as a compromise.

### 6.3 Using hardware you already own

Any two GPUs with a real bandwidth gap work — the experiment is about
heterogeneity, not about rented cards. But two constraints decide whether a given
card can be a *paper* worker, and neither is about speed.

**VRAM floor.** The paper uses Qwen2.5-3B-Instruct in every regime
(`DIO_ClusterComputing.tex:130`, `:576`), which is ~6.2 GB of bf16 weights before
any KV cache. **12 GB is the practical minimum**; 8 GB is tight, 6 GB will not
load. Dropping to Qwen2.5-1.5B for Regime D alone would work technically but breaks
comparability with Regimes A–C, and a reviewer will ask why one regime changed
models.

**Thermal stability matters more than peak speed.** The whole method estimates a
per-token slope `s` that is assumed stationary within a run. A card that throttles
partway through a 3-hour campaign does not add noise that averages out — it adds a
*trend*, and a drifting slope corrupts D2 (slope recovery) specifically, which is
the cell carrying the paper's central claim. This rules out laptop GPUs regardless
of their spec sheet: sustained decode in a laptop chassis throttles, and it
throttles in a way that correlates with elapsed time.

Workstation and datacentre cards are fine here. An RTX A6000 (48 GB, 768 GB/s) is
a perfectly good fast worker and pairs to >2x against anything from an L4 downward.

**If one worker is on your own network and the other is rented,** the honest cost is
asymmetric jitter: one worker answers over loopback or LAN, the other over the
public internet. A *constant* offset is harmless — it lands in the intercept `b`,
not the slope — but internet jitter is not constant. It is usually still small
against 1–4 s request latencies, so this is a weakness to disclose rather than a
disqualification. Two nodes in one datacentre remains the cleaner design, and
self-hosting also means opening inbound 8000 on a machine you may not control.

### 6.4 Getting the code onto the nodes

**Do not `git clone` for this.** The Regime D harness and this whole `cloud/`
directory are untracked local work — `git ls-files` returns nothing for
`run_regime_d_hetero.py` or `scripts/cloud/`. A clone on the node succeeds and then
fails at runtime with a missing file, which is the worst failure mode available:
you pay for two idle GPUs while debugging it. Copy from the laptop instead, which
also guarantees the nodes run exactly the code you tested.

`dio-serve/` is ~8 MB, small enough to send whole rather than curating a file list.
From **PowerShell on your laptop**, once per node:

```powershell
cd $env:USERPROFILE\OneDrive\Desktop\Go-serve
ssh -i $env:USERPROFILE\.ssh\dio_cloud ubuntu@<NODE_IP> "mkdir -p ~/Go-serve"
scp -i $env:USERPROFILE\.ssh\dio_cloud -r dio-serve ubuntu@<NODE_IP>:~/Go-serve/
```

Both nodes get the same copy. What differs is which scripts each one runs:

| Node | Runs | Installs |
|------|------|----------|
| Fast GPU (A30) | `setup_worker.sh`, `start_vllm.sh` | vLLM + CUDA (~8 GB) |
| Slow GPU (L4), also gateway | both of the above, **then** `run_all.sh` | vLLM, plus `dio-gw-venv` |

The gateway install is genuinely light — DIO's dependencies are fastapi, uvicorn,
httpx, pydantic, typer and rich, no torch — so co-locating it costs the L4 node
seconds, not minutes.

**If `scp -r` is slow or flaky** on a home connection, send an archive instead:

```powershell
tar -czf dio-serve.tgz dio-serve
scp -i $env:USERPROFILE\.ssh\dio_cloud dio-serve.tgz ubuntu@<NODE_IP>:~/
ssh -i $env:USERPROFILE\.ssh\dio_cloud ubuntu@<NODE_IP> "mkdir -p ~/Go-serve && tar -xzf ~/dio-serve.tgz -C ~/Go-serve"
```

Windows 11 ships `tar`, `ssh` and `scp` — nothing to install.

**Verify before you start anything expensive**, on each node:

```bash
ls ~/Go-serve/dio-serve/scripts/run_regime_d_hetero.py \
   ~/Go-serve/dio-serve/scripts/cloud/setup_worker.sh
```

Two paths echoed back means you are good. A `No such file` here costs nothing;
finding out later costs ₹2.3/minute.

## 7. Tighten the port-8000 rule

Now that the gateway node exists, go back to **Network → Security Groups →
dio-regime-d → Inbound Rules** and edit the port-8000 rule:

- **Source IPv4 / Network:** `Custom (Manual)` → `<dio-l4 PRIVATE IP>/32` if you
  have a VPC, or `<dio-l4 PUBLIC IP>/32` if you do not

The L4 node is the gateway, so its address is the only one that ever needs to reach
the A30's port 8000. (The reverse rule is unnecessary: the L4's own vLLM is reached
over loopback, never across the network.)

If you already set this rule to the VPC CIDR in §5, you can skip this section —
`10.0.0.0/8` is private and the traffic never leaves E2E's network. Narrowing it to
the gateway's single address is still slightly better and costs you one edit.

This means only the gateway can reach the inference endpoints. Leaving port 8000
open to `0.0.0.0/0` would put an unauthenticated LLM on the public internet for
anyone to use at your expense.

Verify from your laptop that it is actually closed:

```powershell
Test-NetConnection -ComputerName <L4_PUBLIC_IP> -Port 8000
```

`TcpTestSucceeded : False` from your laptop is the result you want — combined with
`preflight.sh` passing on the gateway, that proves the port is reachable by the
gateway and nobody else.

## 8. Set up the machines

### 8.1 Connect

From your laptop, one PowerShell window per node (Windows 10+ has SSH built in):

```powershell
ssh -i $env:USERPROFILE\.ssh\dio_cloud ubuntu@<PUBLIC_IP>
```

First connection asks `Are you sure you want to continue connecting?` → type `yes`.

If it rejects you with `Permission denied (publickey)`, the username is wrong — try
`root@` instead of `ubuntu@`; E2E images vary. The console's node detail page shows
the correct default user.

### 8.2 Copy the code up (from your laptop, new window)

Per §6.4 — the whole `dio-serve` tree, not just the `cloud` folder, because the
gateway node needs `run_regime_d_hetero.py` and the package itself:

```powershell
cd $env:USERPROFILE\OneDrive\Desktop\Go-serve
ssh -i $env:USERPROFILE\.ssh\dio_cloud ubuntu@<L4_PUBLIC_IP> "mkdir -p ~/Go-serve"
scp -i $env:USERPROFILE\.ssh\dio_cloud -r dio-serve ubuntu@<L4_PUBLIC_IP>:~/Go-serve/
ssh -i $env:USERPROFILE\.ssh\dio_cloud ubuntu@<A30_PUBLIC_IP> "mkdir -p ~/Go-serve"
scp -i $env:USERPROFILE\.ssh\dio_cloud -r dio-serve ubuntu@<A30_PUBLIC_IP>:~/Go-serve/
```

### 8.3 Both GPU nodes (run on each, they can go in parallel)

```bash
cd ~/Go-serve/dio-serve/scripts/cloud
bash setup_worker.sh          # 10-20 min: installs vLLM, downloads the model
bash start_vllm.sh 8000       # 1-3 min: starts the server, waits for readiness
```

`start_vllm.sh` prints `READY` plus the GPU name and a sample of the Prometheus
metrics DIO scrapes. If it prints a traceback instead, the last 40 log lines are
shown — the usual cause is out-of-memory, fixed by re-running with a lower
fraction: `GPU_UTIL=0.75 bash start_vllm.sh 8000`.

**Record the exact GPU string it prints** (e.g. `NVIDIA L4, 23034 MiB`). That goes
in the paper's hardware table.

### 8.4 Gateway (the slow GPU node — see §6.2)

Stay on the **L4** node; it already has the code from §8.2 and is already serving
vLLM on :8000. All that is missing is tmux:

```bash
sudo apt-get install -y tmux jq
tmux new -s dio               # ALWAYS work inside tmux for the campaign
```

Inside tmux, verify both workers are reachable **before** spending anything. The
local worker is loopback, the remote one is its private IP:

```bash
bash ~/Go-serve/dio-serve/scripts/cloud/preflight.sh \
  "l4=http://127.0.0.1:8000,a30=http://<A30_PRIVATE_IP>:8000"
```

This checks reachability, that `/metrics` is alive, that both workers serve the
*same* model, and that a real completion succeeds. It must print
`PREFLIGHT PASSED`. If it fails, fix it now — every minute of debugging with both
GPUs up costs ₹2.3.

### 8.5 Run the campaign

```bash
bash ~/Go-serve/dio-serve/scripts/cloud/run_all.sh \
  "l4=http://127.0.0.1:8000,a30=http://<A30_PRIVATE_IP>:8000" \
  "l4=24000,a30=24000"
```

The second argument is **declared VRAM in MB per worker**, and it must match the
real cards. It is what the gateway is told, not what it measures. Getting it wrong
is the one input error that biases routing silently rather than crashing.

Roughly 2.5–3.5 hours for the default 10 seeds × 40 requests across the three
cells. See §9.1 for where that time goes and how to shorten it.

**tmux survival:** detach with `Ctrl-b` then `d`. Your SSH can drop, your laptop can
sleep — the run continues. Reattach with `ssh` back in then `tmux attach -t dio`.

### 8.6 Collect and shut down

From your laptop:

```bash
bash dio-serve/scripts/cloud/collect.sh <L4_PUBLIC_IP>
```

The gateway is the L4 node, so that is the IP to pull from.

Then **immediately** go to **Compute → Nodes** and **delete both nodes**.
Not "stop" — **delete**. A stopped node can still bill for its attached storage.

Check **Billing** afterwards to confirm the hourly charge has stopped.

---

## 9. Money

Two GPU nodes, gateway co-located on the L4 (§6.2), billed for 5 hours:

| Item | Rate | ~Hours | Cost |
|------|------|--------|------|
| A30 24GB | ₹90/hr | 5 | ₹450 |
| L4 24GB (also runs the gateway) | ₹49/hr | 5 | ₹245 |
| **Subtotal** | | | **₹695** |
| +18% GST | | | ₹125 |
| **Total per campaign** | | | **~₹820** |

If you substitute A100-40 for the A30, add ₹89/hr — about ₹525 more including GST.

### 9.0 When billing starts and stops

**The clock starts when the node reaches Running, not when you start using it.** An
idle GPU bills exactly like a busy one. Consequences worth planning around:

- The ~25 minutes of `setup_worker.sh` (torch/CUDA wheels, model download) is billed
  at the full GPU rate. That is already in the 3.5–4.5 h estimate; it is not
  avoidable, only parallelisable — which is why both GPU nodes are set up at the
  same time rather than one after the other.
- Assume **partial hours round up**. Deleting at 3 h 10 m may bill 4 hours. Do not
  design around finishing 5 minutes under an hour boundary.
- **Only deleting stops the charge.** Powering a node off leaves storage allocated
  and still billable on most providers, E2E included.

**Create the L4 first.** Launch `dio-l4` (₹49/hr) alone, get through §8.2-8.3 on
it, and check **Billing** to confirm the burn rate looks the way you expect. You
learn how E2E presents charges while the meter runs at forty-nine rupees an hour
rather than a hundred and thirty-nine. Only then create the A30 — which also means
the expensive card is idle for less of the setup window.

Your ₹3,000 top-up at ~₹139/hr is roughly 21 hours of runway against a 4-hour
campaign, so credit exhaustion mid-run is not a realistic risk — but note that if
prepaid credit *did* hit zero, E2E can suspend the nodes and the run would die with
them. Do not let the balance drift down near one campaign's cost.

### 9.1 Where the time actually goes

The harness sends requests **one at a time** (`run_regime_d_hetero.py:263`), by
design: concurrent load would let queueing dominate and mask the per-token slope
the experiment is trying to measure. So runtime is just request count x per-request
latency, and it is predictable.

Default campaign = 2,600 requests:

| Cell | Work | Requests |
|------|------|----------|
| D1 | 4 strategies x 10 seeds x 40 req | 1,600 |
| D2 | 10 seeds x 40 req (round-robin) | 400 |
| D3 | 3 lengths x 5 seeds x 40 req | 600 |

Mean output over the mixed 32/64/128/256 set is ~120 tokens. Qwen2.5-3B in bf16 is
~6 GB of weights, so decode throughput is roughly bandwidth / weights: ~30-40 tok/s
on the L4, ~90-110 on the A30. That is ~3-4 s per request on the slow worker and
~1.5 s on the fast one, plus ~65 gateway restarts at ~4 s each.

| Phase | Wall-clock |
|-------|-----------|
| Node create + boot (both) | 5-10 min |
| `setup_worker.sh` (vLLM + model download, both nodes in parallel) | 15-25 min |
| `start_vllm.sh` + `preflight.sh` | 5 min |
| **Campaign (`run_all.sh`)** | **2.5-3.5 h** |
| `collect.sh` + delete nodes | 5 min |
| **Total GPU-billable** | **~3.5-4.5 h** |

Budget 5 hours of billing and ₹3,000 of credit, so one failed attempt doesn't
strand you.

**To shorten it:** `--seeds 5` roughly halves D1 and D2 (~1.5 h). Drop `--d3` to
save ~25 min, though D3 is the cell that demonstrates the identifiability failure
honestly, so I would keep it and cut seeds instead. Do not raise concurrency to go
faster — that changes what is being measured.

**Rules that save money:**

- Delete nodes the moment `collect.sh` succeeds. This is the only thing that
  actually stops billing.
- Never leave GPUs running overnight "to finish tomorrow" — that is ₹1,700 of idle
  A30+L4, or ₹5,500 if you rented an A100.
- Do all setup and debugging on the ₹3/hr gateway where possible.
- If you must pause, delete the GPU nodes and re-run `setup_worker.sh` later. The
  20 minutes of reinstall is far cheaper than idle GPU hours.
- Set a **billing alert** in the console if E2E offers one.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Permission denied (publickey)` | wrong username | try `root@` instead of `ubuntu@`; check the node page |
| preflight: `cannot GET /v1/models` | port 8000 closed, or vLLM bound to localhost | check the security-group rule source is the gateway's **private** IP; confirm `start_vllm.sh` used `--host 0.0.0.0` |
| preflight: `/metrics unreachable` | vLLM too old, or metrics disabled | `pip install -U vllm` on the worker |
| preflight: `workers serve DIFFERENT models` | typo'd model name on one node | re-run `setup_worker.sh` with the identical string on both |
| `start_vllm.sh` OOM | KV cache too large for the card | `GPU_UTIL=0.75 bash start_vllm.sh 8000` |
| vLLM never becomes ready | still downloading weights | check `tail -f ~/vllm_8000.log`; `setup_worker.sh` should have pre-cached them |
| Harness hangs at "backend not reachable" | preflight was skipped | Ctrl-C, run preflight, fix, restart |
| Run died when laptop slept | not inside tmux | always `tmux new -s dio` first |
| routing fractions are 50/50 under NLMS | gateway never learned a difference | check `summary.json` → `learned` slopes; if `updates` is near zero the feedback path is broken |

---

## 11. What the results mean

`results_regime_d/paper_snippets.md` is written for you. Three things to read:

**D1** — p99 latency per strategy plus routing fractions. NLMS should send
noticeably more than 50% of traffic to the faster SKU and beat round-robin on p99. The
`wins N/M` count matters as much as the mean: 8/10 seeds winning is a result,
a big mean carried by one seed is not.

**D2** — the learned NLMS slope per SKU next to an offline OLS fit on the same
observed data. This is the honest version of "DIO discovers which card is faster":
OLS is the ground truth, and the comparison shows whether the online learner
recovers it. Expect the learned slope to be *shrunk* toward zero relative to OLS —
NLMS is a tracking filter, not an unbiased estimator. What must hold is the
**ordering** and roughly the ratio, because ranking is all the scheduler needs.

**D3** — the same measurement at fixed `max_tokens` of 32 / 128 / 256. At short
decode lengths the token feature barely varies, so `y ≈ sN + b` is ill-conditioned
and the split between slope and intercept is arbitrary. `degenerate_ols_seeds > 0`
is that failure being caught explicitly. This cell is why D1 and D2 use *mixed*
output lengths, and it is worth reporting rather than hiding.

---

## 12. From results to submitted paper

Run these on your laptop after `collect.sh`, in this order. Steps 1–2 are
mechanical; step 3 is the only one that needs you to write prose.

Both scripts **refuse to run on a `--mock` summary** — fixture numbers cannot
reach the paper even by accident. If you see that refusal, you pointed them at a
local test run instead of the cloud results.

### 12.1 Fill the two tables

```powershell
cd $env:USERPROFILE\OneDrive\Desktop\Go-serve\dio-serve
python scripts\fill_regime_d_tables.py `
  --summary results_regime_d\summary.json `
  --tex ..\paper_drafts_latex\cluster_computing_submission\DIO_ClusterComputing.tex `
  --dry-run
```

Read the printed rows first. If they look right, re-run **without** `--dry-run`;
it saves `DIO_ClusterComputing.tex.bak` before writing and can be re-run safely.

### 12.2 Draw the figure

```powershell
python scripts\plot_regime_d.py `
  --summary results_regime_d\summary.json `
  --out ..\paper_drafts_latex\cluster_computing_submission\fig_regime_d.png
```

Then add it to the `.tex` inside `\subsection{Regime D...}`, after Table
`tab:slopeD`:

```latex
\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{fig_regime_d.png}
\caption{Regime D. (a) Online NLMS per-token slope against an offline OLS fit on
the same observed pairs, per SKU; the learned slope is shrunk toward zero, as
expected of a tracking filter, but the ordering the scheduler consumes is
preserved. (b) p99 by strategy, mean $\pm$ s.d.\ over seeds, annotated with the
paired win count against RR.}
\label{fig:regimeD}
\end{figure}
```

### 12.3 Write the Results paragraph

Replace the `\textbf{Results.} TBD ...` line (search for `TBD`). State only what
the numbers support, and in this order:

1. **Did NLMS route asymmetrically?** Quote `route_frac` for the fast SKU. Around
   50/50 means it never learned the difference — report that instead of a win,
   and check `learned.updates` before assuming the result is real.
2. **Did p99 improve, and how consistently?** Quote mean ± s.d. *and* the
   `wins N/M` count. If the error bar overlaps RR, say the improvement is not
   separable at this sample size.
3. **Did the learned slope recover the offline fit?** Ordering and rough ratio
   are the claim. Expect shrinkage toward zero and say so — NLMS is a tracking
   filter, not an unbiased estimator.
4. **D3:** if `ols_degenerate_seeds > 0`, report it as the identifiability
   failure being detected, which is why D1/D2 mix output lengths.

Then confirm no `TBD` survives:

```powershell
Select-String -Path ..\paper_drafts_latex\cluster_computing_submission\DIO_ClusterComputing.tex -Pattern '\bTBD\b'
```

### 12.4 Update the cover letter

`COVER_LETTER_ClusterComputing.txt` currently says the evaluation is *"two T4
GPUs and a 3B model, not an A100/H100 fleet"* — Regime D makes that false. Edit
the **Scope and limitations** paragraph to name the SKUs you actually rented, and
add one line to **Empirical results** for the Regime D outcome. Keep the honest
framing: two workers, one model size.

### 12.5 Recompile and check

```powershell
cd ..\paper_drafts_latex\cluster_computing_submission
pdflatex -interaction=nonstopmode DIO_ClusterComputing.tex
bibtex DIO_ClusterComputing
pdflatex -interaction=nonstopmode DIO_ClusterComputing.tex
pdflatex -interaction=nonstopmode DIO_ClusterComputing.tex
```

Three passes are not superstition: the first resolves labels, bibtex builds the
bibliography, and the last two settle references and page numbers. Then confirm
`Output written ... (N pages)` and that nothing reports an undefined reference or
citation.

### 12.6 Submit

Cluster Computing uses **Editorial Manager**
(`https://www.editorialmanager.com/clus/`). Upload:

| Item | File |
|------|------|
| Manuscript | `DIO_ClusterComputing.pdf` |
| Source | `.tex`, `sn-bibliography.bib`, `sn-jnl.cls`, all `fig_*.png` |
| Cover letter | `COVER_LETTER_ClusterComputing.txt` |

Declarations the form will ask for: no competing interests; no external funding
unless you have some to declare; all data and code in the public repository.
Keep the raw `results_regime_d/` directory — reviewers ask for it.
