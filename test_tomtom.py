import requests

# NOTE: I cannot sign up for an API key as an AI agent. 
# You can replace this with your actual key to verify.
API_KEY = "YOUR_TOMTOM_API_KEY_HERE"

def test_tomtom_traffic():
    # Lagos coordinates for testing
    coords = {
        "Ikeja": "6.6018,3.3515",
        "Yaba": "6.5056,3.3744",
        "Surulere": "6.4950,3.3444"
    }

    print("Testing TomTom Traffic Flow API for Lagos...")
    for name, point in coords.items():
        # Using Traffic Flow Relative API
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={point}&key={API_KEY}"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                flow_data = data.get("flowSegmentData", {})
                current_speed = flow_data.get("currentSpeed")
                free_flow_speed = flow_data.get("freeFlowSpeed")
                print(f"[{name}] SUCCESS - Current Speed: {current_speed} km/h, Free Flow: {free_flow_speed} km/h")
            else:
                print(f"[{name}] FAILED - HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[{name}] ERROR: {str(e)}")

if __name__ == "__main__":
    test_tomtom_traffic()
