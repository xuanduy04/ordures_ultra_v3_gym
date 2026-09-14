
from pydantic import Field

from aviary.envs.hotpotqa import HotPotQADataset, HotPotQAEnv
from resources_servers.aviary.app import AviaryResourcesServer


class HotPotQAResourcesServer(AviaryResourcesServer[HotPotQAEnv, HotPotQADataset]):
    dataset: HotPotQADataset = Field(default_factory=lambda: HotPotQADataset(split="train"))


if __name__ == "__main__":
    HotPotQAResourcesServer.run_webserver()
