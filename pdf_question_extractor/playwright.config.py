"""
Playwright configuration for E2E testing
"""
from playwright.sync_api import Playwright
import os

# Configuration
class PlaywrightConfig:
    # Base URL for tests
    base_url = os.getenv("TEST_BASE_URL", "http://localhost:8000")
    
    # Browser settings
    browser_options = {
        "headless": os.getenv("HEADLESS", "true").lower() == "true",
        "slow_mo": int(os.getenv("SLOW_MO", "0")),  # Slow down actions by X ms
    }
    
    # Context settings
    context_options = {
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
        "locale": "en-US",
        "timezone_id": "America/New_York",
    }
    
    # Test settings
    test_timeout = 30000  # 30 seconds per test
    expect_timeout = 5000  # 5 seconds for expect assertions
    
    # Screenshot settings
    screenshot_options = {
        "full_page": True,
        "type": "png",
    }
    
    # Video settings (for debugging)
    video_options = {
        "size": {"width": 1280, "height": 720},
    }
    
    # Browsers to test
    browsers = ["chromium", "firefox", "webkit"]  # webkit = Safari
    
    # Retry settings
    retry_times = 2
    
    # Parallel execution
    workers = 3  # Number of parallel workers

# pytest-playwright configuration
def pytest_configure(config):
    """Configure pytest-playwright"""
    config.option.base_url = PlaywrightConfig.base_url
    config.option.headed = not PlaywrightConfig.browser_options["headless"]
    config.option.slowmo = PlaywrightConfig.browser_options["slow_mo"]
    config.option.screenshot = "only-on-failure"
    config.option.video = "retain-on-failure"

# Browser launch options
def browser_context_args(browser_context_args):
    """Extend default browser context arguments"""
    return {
        **browser_context_args,
        **PlaywrightConfig.context_options,
        "record_video_dir": "./test-results/videos" if os.getenv("RECORD_VIDEO") else None,
    }