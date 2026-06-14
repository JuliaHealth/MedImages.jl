import asyncio
import os
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 1800})
        
        file_path = f"file://{os.path.abspath('article/figures/study_flowchart.html')}"
        print(f"Opening {file_path}")
        
        await page.goto(file_path)
        
        print("Waiting for MathJax to render...")
        await page.wait_for_timeout(5000)
        
        out_path = 'article/figures/study_flowchart.png'
        await page.screenshot(path=out_path, full_page=True)
        print(f"Successfully saved to {out_path}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
