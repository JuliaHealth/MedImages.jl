import os
from playwright.sync_api import sync_playwright

def render_html_to_png(html_file, png_file, width=1200):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a very tall height to start with, then clip to container
        page = browser.new_page(viewport={"width": width, "height": 2000})

        # Get absolute path
        abs_path = "file://" + os.path.abspath(html_file)
        page.goto(abs_path)

        # Wait for all images to load
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000) # Extra buffer for any dynamic styles

        # Screenshot the container element precisely
        container = page.locator('.container')
        if container.count() > 0:
            container.screenshot(path=png_file)
            print(f"Rendered {html_file} -> {png_file}")
        else:
            # Fallback to full page if .container not found
            page.screenshot(path=png_file, full_page=True)
            print(f"Rendered {html_file} (Full Page) -> {png_file}")
            
        browser.close()

if __name__ == "__main__":
    # Ensure clinical assets exist (should be there from previous step)
    files = [
        ('elsarticle/figures_new/challenge_1.html', 'elsarticle/figures_new/challenge_1.png', 1100),
        ('elsarticle/figures_new/challenge_2.html', 'elsarticle/figures_new/challenge_2.png', 1100),
        ('elsarticle/figures_new/challenge_3.html', 'elsarticle/figures_new/challenge_3.png', 1100),
        ('elsarticle/figures_new/challenge_4.html', 'elsarticle/figures_new/challenge_4.png', 1100),
        ('elsarticle/figures_new/dosimetry_experiment.html', 'elsarticle/figures_new/dosimetry_experiment.png', 1200)
    ]

    for html, png, width in files:
        if os.path.exists(html):
            render_html_to_png(html, png, width)
        else:
            print(f"Warning: {html} not found")
