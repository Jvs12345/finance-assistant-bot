import socket
import sys
import psutil
import urllib.request
import urllib.error

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def check_http_endpoint(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status, response.read().decode('utf-8')[:100]
    except urllib.error.URLError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)

def main():
    print("=== Connection Diagnostics ===")
    
    ports_to_check = [8000, 8080, 11434]
    
    # 1. Check Ports
    print("\n[1] Checking TCP Ports:")
    for port in ports_to_check:
        is_open = check_port('localhost', port)
        status = "OPEN/LISTENING" if is_open else "CLOSED"
        print(f"  Port {port}: {status}")

    # 2. Check Processes
    print("\n[2] Checking Related Processes:")
    targets = ['python', 'uvicorn', 'ollama']
    found = False
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            name = proc.info['name'].lower()
            cmd = proc.info['cmdline'] or []
            cmd_str = ' '.join(cmd).lower()
            
            for t in targets:
                if t in name or any(t in c for c in cmd_str):
                    print(f"  Found: {name} (PID: {proc.pid}) - {cmd_str[:100]}...")
                    found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if not found:
        print("  No obvious python/uvicorn/ollama processes found running.")

    # 3. HTTP Health Checks
    print("\n[3] HTTP Health Checks:")
    endpoints = [
        "http://localhost:8080/health",
        "http://localhost:8000/health",
        "http://localhost:11434/"
    ]
    
    for url in endpoints:
        status, response = check_http_endpoint(url)
        print(f"  {url}: {status} - {response}")

if __name__ == "__main__":
    main()
