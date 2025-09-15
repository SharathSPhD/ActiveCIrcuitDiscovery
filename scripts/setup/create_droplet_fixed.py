#!/usr/bin/env python3
"""
Create GPU droplet from golden snapshot - FIXED VERSION
- Reads token from digit1.txt
- Simplified status checking that doesn't hang
- Just waits reasonable time for snapshot to become active
- Takes snapshot name as user input argument
"""
import os
import json
import requests
import time
import sys
import argparse
from datetime import datetime
from pathlib import Path

def get_api_token():
    """Get API token from digit1.txt file"""
    try:
        token_file = Path(__file__).parent / "digit1.txt"
        if token_file.exists():
            token = token_file.read_text().strip()
            print(f"✅ Token loaded from digit1.txt")
            return token
        else:
            print("❌ digit1.txt not found")
            return None
    except Exception as e:
        print(f"❌ Error reading token: {e}")
        return None

def get_snapshot_id_by_name(snapshot_name, token):
    """Get snapshot ID by name"""
    url = "https://api.digitalocean.com/v2/snapshots"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            snapshots = response.json()['snapshots']
            for snapshot in snapshots:
                if snapshot['name'] == snapshot_name:
                    print(f"✅ Found snapshot: {snapshot_name} (ID: {snapshot['id']})")
                    return snapshot['id']
            
            # If exact match not found, show available snapshots
            print(f"❌ Snapshot '{snapshot_name}' not found")
            print("📋 Available snapshots:")
            for snapshot in snapshots[:10]:  # Show first 10
                print(f"   - {snapshot['name']} (ID: {snapshot['id']})")
            return None
        else:
            print(f"❌ Failed to fetch snapshots: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching snapshots: {e}")
        return None

def create_gpu_droplet_from_snapshot(snapshot_name):
    # Get API token from file
    token = get_api_token()
    if not token:
        return None
    
    # Get snapshot ID by name
    if snapshot_name.isdigit():
        # If user provided ID directly
        snapshot_id = int(snapshot_name)
        print(f"Using snapshot ID: {snapshot_id}")
    else:
        # Look up snapshot by name
        snapshot_id = get_snapshot_id_by_name(snapshot_name, token)
        if not snapshot_id:
            return None
    
    # API endpoint
    url = "https://api.digitalocean.com/v2/droplets"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Payload using specified snapshot
    payload = {
        "name": f"snapshots-gpu-l40sx1-{datetime.now().strftime('%Y%m%d-%H%M')}",
        "size": "gpu-l40sx1-48gb",  # L40S GPU with 48GB
        "region": "tor1",           # Toronto region
        "image": snapshot_id,       # User-specified snapshot
        "ssh_keys": [48299916],     # SSH key ID
        "tags": ["gpu", "l40s", "snapshot", "activecd"],
        "vpc_uuid": "89ad4747-9fc9-42f3-a25f-180699ed0f43",
        "monitoring": True,
        "ipv6": True,
        "backups": False
    }
    
    print("🚀 Creating GPU droplet from snapshot...")
    print(f"   Name: {payload['name']}")
    print(f"   Size: {payload['size']} (L40S GPU)")
    print(f"   Region: {payload['region']}")
    print(f"   Image ID: {payload['image']} (Snapshot: {snapshot_name})")
    
    try:
        # Make the API request
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 202:  # Accepted
            droplet_data = response.json()
            droplet = droplet_data['droplet']
            
            print("✅ Droplet creation initiated successfully!")
            print(f"   Droplet ID: {droplet['id']}")
            print(f"   Name: {droplet['name']}")
            print(f"   Status: {droplet['status']}")
            print(f"   Region: {droplet['region']['name']}")
            print(f"   Size: {droplet['size']['slug']}")
            
            return {
                'droplet_id': droplet['id'],
                'name': droplet['name'],
                'status': droplet['status'],
                'token': token
            }
            
        else:
            print(f"❌ API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
                if 'errors' in error_data:
                    for error in error_data['errors']:
                        print(f"   - {error}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def check_droplet_status_simple(droplet_id, token):
    """Simple status check with timeout protection"""
    url = f"https://api.digitalocean.com/v2/droplets/{droplet_id}"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)  # Short timeout
        if response.status_code == 200:
            droplet = response.json()['droplet']
            status = droplet['status']
            print(f"📊 Current Status: {status}")
            
            if status == 'active':
                networks = droplet['networks']['v4']
                public_ip = next((net['ip_address'] for net in networks if net['type'] == 'public'), None)
                if public_ip:
                    print(f"🌐 Public IP: {public_ip}")
                    return public_ip
            return status
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Status check error (continuing anyway): {e}")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create GPU droplet from DigitalOcean snapshot')
    parser.add_argument('snapshot', nargs='?', 
                       default='ActiveCircuitDiscovery-L40S-Working-15062025',
                       help='Snapshot name or ID (default: ActiveCircuitDiscovery-L40S-Working-15062025)')
    parser.add_argument('--list-snapshots', action='store_true', 
                       help='List available snapshots and exit')
    
    args = parser.parse_args()
    
    # If user wants to list snapshots
    if args.list_snapshots:
        token = get_api_token()
        if token:
            print("📋 Fetching available snapshots...")
            url = "https://api.digitalocean.com/v2/snapshots"
            headers = {"Authorization": f"Bearer {token}"}
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    snapshots = response.json()['snapshots']
                    print(f"Found {len(snapshots)} snapshots:")
                    for snapshot in snapshots:
                        print(f"   - {snapshot['name']} (ID: {snapshot['id']})")
                else:
                    print(f"❌ Failed to fetch snapshots: {response.status_code}")
            except Exception as e:
                print(f"❌ Error: {e}")
        sys.exit(0)
    
    # Create the droplet
    result = create_gpu_droplet_from_snapshot(args.snapshot)
    
    if result:
        print(f"\n✅ Droplet '{result['name']}' created successfully!")
        print(f"   ID: {result['droplet_id']}")
        
        # Wait for snapshot to become active (usually 2-3 minutes)
        print("\n⏳ Waiting for snapshot to become active...")
        print("   Snapshots typically become active in 2-3 minutes")
        
        # Check status a few times with delays
        for attempt in range(6):  # Check 6 times over ~3 minutes
            if attempt > 0:
                print(f"   Attempt {attempt + 1}/6...")
                time.sleep(30)  # Wait 30 seconds between checks
            
            status = check_droplet_status_simple(result['droplet_id'], result['token'])
            
            if isinstance(status, str) and '.' in status:  # Got IP address
                print(f"\n🎉 Droplet is ACTIVE!")
                print(f"🚀 Ready to connect: ssh ubuntu@{status}")
                print("\n📋 Next steps:")
                print("   1. SSH into the droplet:")
                print(f"      ssh ubuntu@{status}")
                print("   2. Test GPU: nvidia-smi")
                print("   3. Test PyTorch: python3 -c 'import torch; print(torch.cuda.is_available())'")
                print("   4. Navigate to project: cd ActiveCircuitDiscovery")
                print("   5. Run experiment: python run_complete_experiment.py")
                break
            elif status == 'active':
                print("   ✅ Droplet is active, getting IP...")
                continue
            else:
                print(f"   Status: {status}, waiting...")
        else:
            print(f"\n⚠️ Droplet may still be starting up")
            print(f"   Check manually: https://cloud.digitalocean.com/droplets/{result['droplet_id']}")
            print(f"   Or try: ssh ubuntu@<IP_ADDRESS>")
    else:
        print("\n❌ Failed to create droplet")