import kagglehub
import os
import shutil
import torch
import json

class KaggleHubManager:
    """
    Manages model interactions with the Kaggle Model Hub using kagglehub.
    """
    def __init__(self, model_handle=None):
        self.model_handle = model_handle

    def download_model(self, handle=None):
        handle = handle or self.model_handle
        if not handle:
            raise ValueError("No model handle provided.")

        print(f"Downloading model from Kaggle Hub: {handle}...")
        path = kagglehub.model_download(handle)
        print(f"Model downloaded to: {path}")
        return path

    def upload_model_version(self, handle, local_model_dir, version_notes="New model version"):
        """
        Uploads a new version of a model to Kaggle Model Hub.
        handle: 'owner/model/framework/variation'
        """
        if not os.path.exists(local_model_dir):
            raise FileNotFoundError(f"Local model directory {local_model_dir} does not exist.")

        print(f"Uploading model version to {handle} from {local_model_dir}...")
        try:
            path = kagglehub.model_upload(handle, local_model_dir, version_notes=version_notes)
            print(f"Model successfully uploaded to {handle}")
            return path
        except Exception as e:
            print(f"Failed to upload model: {e}")
            return None

def save_and_push_to_hub(model, optimizer, epoch, metrics, handle, local_dir='checkpoint'):
    """
    Helper to save a checkpoint locally and push it to Kaggle Hub.
    """
    os.makedirs(local_dir, exist_ok=True)
    checkpoint_path = os.path.join(local_dir, 'model.pt')
    torch_state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(torch_state, checkpoint_path)

    with open(os.path.join(local_dir, 'metadata.json'), 'w') as f:
        json.dump(metrics, f)

    manager = KaggleHubManager()
    manager.upload_model_version(handle, local_dir, version_notes=f"Epoch {epoch} checkpoint")

if __name__ == "__main__":
    pass
