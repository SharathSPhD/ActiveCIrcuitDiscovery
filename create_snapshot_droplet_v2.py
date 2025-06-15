#!/usr/bin/env python3
"""
Create GPU droplet from golden snapshot using corrected DigitalOcean API
Fix: Convert numeric IDs to slugs for modern API compatibility
"""
import os
import json
import requests
from datetime import datetime

def create_gpu_droplet_from_snapshot():
    # Get API token
    token = os.environ.get('DIGITALOCEAN_API_TOKEN')
    if not token:
        print("❌ DIGITALOCEAN_API_TOKEN not found in environment")
        return None
    
    # API endpoint
    url = "https://api.digitalocean.com/v2/droplets"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Fixed payload using modern slug format
    payload = {
        "name": f"snapshots-gpu-l40sx1-{datetime.now().strftime('%Y%m%d-%H%M')}",
        "size": "gpu-l40sx1-48gb",  # L40S GPU with 48GB
        "region": "tor1",           # Toronto region
        "image": 190297195,         # Your golden snapshot image ID
        "ssh_keys": [48299916],     # Your SSH key ID
        "tags": ["gpu", "l40s", "snapshot", "activecd"],
        "vpc_uuid": "89ad4747-9fc9-42f3-a25f-180699ed0f43",
        "monitoring": True,
        "ipv6": True,
        "backups": False
    }
    
    print("🚀 Creating GPU droplet from golden snapshot...")
    print(f"   Name: {payload['name']}")
    print(f"   Size: {payload['size']} (L40S GPU)")
    print(f"   Region: {payload['region']}")
    print(f"   Image ID: {payload['image']} (Golden Snapshot)")
    print(f"   VPC: {payload['vpc_uuid']}")
    
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
            
            # Monitor creation progress
            print("\n🔄 Waiting for droplet to become active...")
            droplet_id = droplet['id']
            
            return {
                'droplet_id': droplet_id,
                'name': droplet['name'],
                'status': droplet['status']
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

def check_droplet_status(droplet_id, token):
    """Check the status of a specific droplet"""
    url = f"https://api.digitalocean.com/v2/droplets/{droplet_id}"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            droplet = response.json()['droplet']
            print(f"📊 Droplet Status: {droplet['status']}")
            
            if droplet['status'] == 'active':
                networks = droplet['networks']['v4']
                public_ip = next((net['ip_address'] for net in networks if net['type'] == 'public'), None)
                if public_ip:
                    print(f"🌐 Public IP: {public_ip}")
                    print(f"🔑 SSH Command: ssh ubuntu@{public_ip}")
                    return public_ip
            return droplet['status']
        else:
            print(f"❌ Failed to check status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None

def wait_for_droplet_active(droplet_id, token, max_wait=300):
    """Wait for droplet to become active"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status = check_droplet_status(droplet_id, token)
        if status == 'active':
            return True
        elif isinstance(status, str) and status not in ['new', 'active']:
            print(f"⚠️ Unexpected status: {status}")
        
        print("⏳ Still waiting...")
        time.sleep(10)
    
    print(f"⏰ Timeout after {max_wait} seconds")
    return False

if __name__ == "__main__":
    # Create the droplet
    result = create_gpu_droplet_from_snapshot()
    
    if result:
        print(f"\n✅ Droplet '{result['name']}' created successfully!")
        print(f"   ID: {result['droplet_id']}")
        
        # Wait for droplet to become active
        token = os.environ.get('DIGITALOCEAN_API_TOKEN')
        if wait_for_droplet_active(result['droplet_id'], token):
            print("\n🎉 Droplet is now ACTIVE and ready for use!")
            final_status = check_droplet_status(result['droplet_id'], token)
            if isinstance(final_status, str) and '.' in final_status:  # IP address
                print(f"\n🚀 Ready to connect: ssh ubuntu@{final_status}")
                print("\n📋 Next steps:")
                print("   1. SSH into the droplet")
                print("   2. Test GPU: nvidia-smi")
                print("   3. Test PyTorch: python3 -c 'import torch; print(torch.cuda.is_available())'")
                print("   4. Navigate to project: cd /home/ubuntu/project")
        else:
            print("\n⚠️ Droplet took longer than expected to become active")
            print(f"   Check manually: https://cloud.digitalocean.com/droplets/{result['droplet_id']}")
    else:
        print("\n❌ Failed to create droplet")