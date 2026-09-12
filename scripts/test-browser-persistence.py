#!/usr/bin/env python3
"""Persistent Playwright contexts retain the shared pre-teardown recorder."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from playwright import sync_api
from browser_evidence import FailureEvidence, recorded_playwright


class PersistentEvidenceTests(unittest.TestCase):
    def test_launch_registers_context_and_restores_hook(self):
        context = Mock(pages=[])
        evidence = FailureEvidence()
        owner = object()
        launch = Mock(return_value=context)
        @contextmanager
        def factory():
            yield SimpleNamespace()
        with patch.object(sync_api.BrowserType, "launch_persistent_context", launch):
            with recorded_playwright(evidence, factory):
                result = sync_api.BrowserType.launch_persistent_context(owner, "profile", headless=True)
                self.assertIs(result, context)
                self.assertIn(context, evidence.contexts)
                context.tracing.start.assert_called_once()
                launch.assert_called_once_with(owner, "profile", headless=True)
            self.assertIs(sync_api.BrowserType.launch_persistent_context, launch)

    def test_failure_is_captured_with_live_persistent_context(self):
        context = Mock(pages=[])
        evidence = FailureEvidence()
        original = AssertionError("persistent assertion")
        alive = [True]
        observed = []
        def capture(error):
            self.assertIs(error, original)
            observed.append(alive[0] and context in evidence.contexts)
        @contextmanager
        def factory():
            try:
                yield SimpleNamespace()
            finally:
                alive[0] = False
        with patch.object(sync_api.BrowserType, "launch_persistent_context", Mock(return_value=context)), \
                patch.object(evidence, "capture", side_effect=capture):
            with self.assertRaises(AssertionError) as raised:
                with recorded_playwright(evidence, factory):
                    sync_api.BrowserType.launch_persistent_context(object(), "profile")
                    raise original
        self.assertIs(raised.exception, original)
        self.assertEqual(observed, [True])

    def test_launch_failure_preserves_error_and_restores_hook(self):
        evidence = Mock()
        error = RuntimeError("launch failed")
        launch = Mock(side_effect=error)
        @contextmanager
        def factory():
            yield SimpleNamespace()
        with patch.object(sync_api.BrowserType, "launch_persistent_context", launch):
            with self.assertRaises(RuntimeError) as raised:
                with recorded_playwright(evidence, factory):
                    sync_api.BrowserType.launch_persistent_context(object(), "profile")
            self.assertIs(sync_api.BrowserType.launch_persistent_context, launch)
        self.assertIs(raised.exception, error)
        evidence.capture.assert_called_once_with(error)
        evidence.observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
