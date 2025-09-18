#!/usr/bin/env python3
"""
Circuit-Tracer Webpage Capture
Captures screenshots of the running circuit-tracer server visualization
"""

import asyncio
from playwright.async_api import async_playwright
import os
from pathlib import Path

async def capture_circuit_tracer_page():
    """Capture the circuit-tracer server webpage as PNG"""
    print("🖼️  Starting webpage capture...")

    # Create output directory
    output_dir = Path("visualizations/circuit_tracer_native")
    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        print("🚀 Launching headless browser...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Set viewport size for better screenshot
        await page.set_viewport_size({"width": 1920, "height": 1080})

        try:
            # Navigate to circuit-tracer server
            url = "http://127.0.0.1:8081"
            print(f"🌐 Navigating to {url}...")
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for any dynamic content to load
            await page.wait_for_timeout(3000)

            # Capture full page screenshot
            screenshot_path = output_dir / "circuit_tracer_server_capture.png"
            print(f"📸 Capturing screenshot to {screenshot_path}...")
            await page.screenshot(
                path=str(screenshot_path),
                full_page=True,
                quality=95
            )

            # Get page title and content info
            title = await page.title()
            content = await page.content()

            print(f"✅ Screenshot captured successfully!")
            print(f"   📄 Page title: {title}")
            print(f"   📊 HTML size: {len(content)} characters")
            print(f"   💾 Saved to: {screenshot_path}")

            # Check if we can find circuit-tracer specific elements
            try:
                # Look for common circuit-tracer elements
                nodes = await page.query_selector_all('[data-node-id]')
                edges = await page.query_selector_all('.edge')
                features = await page.query_selector_all('.feature')

                print(f"   🔍 Found {len(nodes)} nodes, {len(edges)} edges, {len(features)} features")

            except Exception as e:
                print(f"   ⚠️  Could not analyze page elements: {e}")

            return screenshot_path

        except Exception as e:
            print(f"❌ Error capturing page: {e}")

            # Try to capture whatever we can see
            fallback_path = output_dir / "circuit_tracer_fallback_capture.png"
            try:
                await page.screenshot(path=str(fallback_path))
                print(f"📸 Fallback screenshot saved to {fallback_path}")
                return fallback_path
            except:
                print("❌ Could not capture fallback screenshot")
                return None

        finally:
            await browser.close()

async def main():
    print("🔬 CIRCUIT-TRACER WEBPAGE CAPTURE")
    print("=" * 50)

    try:
        screenshot_path = await capture_circuit_tracer_page()

        if screenshot_path:
            print(f"\n🎉 Webpage captured successfully!")
            print(f"📁 Screenshot: {screenshot_path}")
            print("\n📋 You can now view the circuit-tracer visualization!")
        else:
            print("\n❌ Failed to capture webpage")

    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())