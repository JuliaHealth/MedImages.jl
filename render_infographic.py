from playwright.sync_api import sync_playwright
import os

def render_html_to_png(html_path, output_path):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
            page = browser.new_page()

            # Convert file path to file:// URL
            file_url = f"file://{os.path.abspath(html_path)}"
            page.goto(file_url)

            # Select the container to take a screenshot of, or full page
            element = page.locator(".container")
            element.screenshot(path=output_path)

            browser.close()
            print(f"Successfully rendered {html_path} to {output_path}")
        except Exception as e:
            print(f"Error rendering {html_path}: {e}")

if __name__ == "__main__":
    render_html_to_png("figures/architecture.html", "figures/architecture.png")
