

from resources_servers.ifbench.setup_ifbench import ensure_ifbench


def pytest_configure(config):
    """Clone and set up IFBench before pytest collects any test modules.

    This runs early enough that instructions_registry is on sys.path by the
    time any test file is imported.
    """
    ensure_ifbench()
