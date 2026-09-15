


MAJOR = 0
MINOR = 3
PATCH = 0
PRE_RELEASE = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "nemo_gym"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://github.com/NVIDIA-NeMo/Gym"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Gym"
__download_url__ = "https://github.com/NVIDIA-NeMo/Gym/releases"
__description__ = "NeMo Gym is a library for building reinforcement learning (RL) training environments"
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, pytorch, torch"
