"""Appium framework skeleton: one smoke test on an Android emulator.

THIS IS A SKELETON. It drives Appium's public sample app (ApiDemos), not a Corvane
app, to show the framework wiring: capabilities, driver lifecycle, a page object
and an assertion. The real value of a mobile suite for a smart fan is BLE and
Wi-Fi provisioning, and that needs real hardware (a fan, a 2.4 GHz router, a
physical phone with Bluetooth); emulators cannot pair with BLE peripherals.

Setup (all free):
  1. Android SDK + an emulator (e.g. Pixel, API 34) running: `emulator -avd <name>`
  2. Appium 2:  `npm i -g appium && appium driver install uiautomator2 && appium`
  3. APK: download ApiDemos-debug.apk from https://github.com/appium/android-apidemos/releases
  4. pip install -r requirements-mobile.txt
  5. APPIUM_SERVER_URL=http://127.0.0.1:4723 ANDROID_APK=/path/ApiDemos-debug.apk pytest tests/mobile

Skipped automatically unless APPIUM_SERVER_URL and ANDROID_APK are set.
"""
import os

import pytest

pytestmark = [
    pytest.mark.mobile,
    pytest.mark.skipif(not (os.getenv("APPIUM_SERVER_URL") and os.getenv("ANDROID_APK")),
                       reason="mobile skeleton: set APPIUM_SERVER_URL and ANDROID_APK to run"),
]


class ApiDemosHome:
    """Page object for the ApiDemos launcher screen."""

    def __init__(self, driver):
        self.driver = driver

    def open_section(self, name: str):
        from appium.webdriver.common.appiumby import AppiumBy

        self.driver.find_element(AppiumBy.ACCESSIBILITY_ID, name).click()

    def has_item(self, name: str) -> bool:
        from appium.webdriver.common.appiumby import AppiumBy

        return bool(self.driver.find_elements(AppiumBy.ACCESSIBILITY_ID, name))


@pytest.fixture
def driver():
    appium = pytest.importorskip("appium.webdriver")
    from appium.options.android import UiAutomator2Options

    options = UiAutomator2Options()
    options.platform_name = "Android"
    options.app = os.environ["ANDROID_APK"]
    options.new_command_timeout = 120
    drv = appium.Remote(os.environ["APPIUM_SERVER_URL"], options=options)
    yield drv
    drv.quit()


def test_app_launches_and_navigates(driver):
    home = ApiDemosHome(driver)
    assert home.has_item("Views"), "launcher list did not render"
    home.open_section("Views")
    assert home.has_item("Animation"), "navigation into Views failed"


@pytest.mark.skip(reason="needs real hardware: a Corvane fan in pairing mode and a physical phone with BLE")
def test_ble_provisioning_placeholder():
    """Would cover: discover fan over BLE, enter the 6-digit PIN, send 2.4 GHz Wi-Fi credentials,
    assert the fan joins Wi-Fi within 30 s, and check E5 when the 3-minute window expires."""
