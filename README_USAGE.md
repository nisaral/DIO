# DIO System - Usage & Testing Guide

This guide explains how to start the Distributed Inference Orchestrator (DIO), access the dashboard, and verify that autoscaling and inference are working correctly.

## 1. Quick Start

1.  Open your terminal (PowerShell or CMD) and navigate to the `DIO` directory.
2.  Run the following command to build and start the system:

    ```powershell
    docker-compose up --build
    ```

3.  Wait for the logs to show:
    *   `DIO Manager gRPC listening at [::]:50052`
    *   `Python Worker listening on port 50053`
    *   `Worker registered successfully!`

## 2. Accessing the Dashboard

Once the system is running, open your web browser and go to:

**http://localhost:8080**

## 3. Running the Tests

Use the buttons on the dashboard to validate different parts of the system.

### **Test A: Latency (Ping)**
*   **Button:** `Run Ping`
*   **What it does:** Sends a single request from the Browser -> Go Manager -> Python Worker -> Go Manager -> Browser.
*   **Success Criteria:** You see `✓ latency test passed!` in the dashboard log box.
*   **What it proves:** The gRPC connection between Go and Python is working.

### **Test B: Spike (Autoscaling)**
*   **Button:** `Simulate Spike`
*   **What it does:** Simulates a burst of traffic to overload the system.
*   **What to watch for:**
    1.  Check your terminal logs for: `[Autoscaler] Demand high. Spawning new Python worker...`
    2.  Open a new terminal and run `docker ps`. You should see **new containers** appearing (e.g., `dio-worker` instances with random names).
*   **What it proves:** The Go Manager successfully talks to the Docker Daemon to scale up infrastructure dynamically.

### **Test C: Token Budget**
*   **Button:** `Check Budget`
*   **What it does:** Sends a request and asks the worker to count tokens.
*   **Success Criteria:** The dashboard log shows a token count (e.g., `Tokens: 7`).
*   **What it proves:** Complex data (like token usage integers) is being correctly serialized and returned from Python to Go.

## 4. Troubleshooting

*   **"Client version is too new"**: This means the Go Docker client version doesn't match your Docker Desktop version. We fixed this by pinning the version to `1.47` in `docker.go`.
*   **"Connection Refused"**: Ensure you are accessing `localhost:8080`. If the worker fails to register, check if `docker-compose` is actually running.
*   **"Protocol message has no tokens_used field"**: This means the Python code is stale. Run `docker-compose up --build` to regenerate the code inside the container.

## 5. Stopping the System

To shut everything down cleanly:

1.  Press `Ctrl+C` in the terminal where the logs are running.
2.  Run this command to remove the containers:
    ```powershell
    docker-compose down
    ```