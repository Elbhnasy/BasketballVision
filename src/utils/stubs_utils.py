import os
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_stub(stub_path:str, obj:object) -> None:
    """
    Save a Python object to disk using PyTorch's optimized serialization.
    Creates necessary directories if they don't exist.
    Args:
        stub_path (str): File path where the object should be saved.
        obj: Any Python object that can be serialized by torch.save.
    """
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    try:
        torch.save(obj, stub_path)
        logger.info(f"Stub saved successfully at {stub_path}")
    except Exception as e:
        logger.error(f"Error saving stub at {stub_path}: {e}")




def read_stub(read_from_stub:bool, stub_path:str, map_location:str="cpu") -> object:
    """
    Read a previously saved Python object from disk using PyTorch's loader.
    Args:
        read_from_stub (bool): Whether to attempt reading from disk.
        stub_path (str): File path where the object was saved.
        map_location (str): Device to map tensors to when loading.
    Returns:
        object: The loaded Python object if successful, None otherwise.
    """
    if not read_from_stub:
        logger.info("Skipping reading from stub as per configuration.")
        return None

    if not os.path.exists(stub_path):
        logger.error(f"Stub file does not exist at {stub_path}")
        return None

    try:
        obj = torch.load(stub_path, map_location=map_location)
        logger.info(f"Stub read successfully from {stub_path}")
        return obj
    except Exception as e:
        logger.error(f"Error reading stub from {stub_path}: {e}")
        return None
    