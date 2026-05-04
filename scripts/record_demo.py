#!/usr/bin/env python3
"""
Screen recording script for Document Search & Retrieval System demo.
Creates a video showing two search examples with the live system.
"""

import asyncio
import time
from playwright.async_api import async_playwright


async def record_demo():
    """Record a demo video of the search system."""

    async with async_playwright() as p:
        # Launch browser with video recording enabled
        browser = await p.chromium.launch(
            headless=True,   # Must use headless in container without X server
            slow_mo=800      # Slow down actions for better visibility (800ms delay)
        )

        # Create context with video recording
        context = await browser.new_context(
            record_video_dir="static/",
            record_video_size={"width": 1280, "height": 720},
            viewport={"width": 1280, "height": 720}
        )

        page = await context.new_page()

        print("🎬 Starting demo recording...")

        # Navigate to the search page
        print("📄 Loading search interface...")
        await page.goto("http://localhost:8000")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)  # Pause to show initial state

        # Demo 1: Search for "User Requirement Specification"
        print("🔍 Demo 1: Searching for 'User Requirement Specification'...")
        search_box = page.locator('input[placeholder*="Search for"]')
        await search_box.click()
        await search_box.fill("User Requirement Specification")
        await asyncio.sleep(1)

        # Click search button
        await page.locator('button:has-text("Search")').click()
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)  # Pause to show results

        # Scroll down to show more results
        await page.evaluate("window.scrollTo(0, 400)")
        await asyncio.sleep(2)

        # Click "Show Full Content" on first result
        print("📖 Expanding full content...")
        await page.locator('button:has-text("Show Full Content")').first.click()
        await asyncio.sleep(3)  # Pause to show full content

        # Scroll down to show the expanded content
        await page.evaluate("window.scrollTo(0, 600)")
        await asyncio.sleep(2)

        # Scroll back up
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)

        # Demo 2: Search for "safety procedures"
        print("🔍 Demo 2: Searching for 'safety procedures'...")
        await search_box.click()
        await search_box.fill("safety procedures")
        await asyncio.sleep(1)

        # Click search button
        await page.locator('button:has-text("Search")').click()
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)  # Pause to show results

        # Scroll down to show more results
        await page.evaluate("window.scrollTo(0, 400)")
        await asyncio.sleep(2)

        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(2)

        print("✅ Demo recording complete!")

        # Close the page and context to finalize the video
        await page.close()
        await context.close()
        await browser.close()

        print("💾 Video saved to static/ directory")
        print("🎉 Demo video generation complete!")


if __name__ == "__main__":
    asyncio.run(record_demo())
