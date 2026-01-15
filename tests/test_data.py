from torch.utils.data import Dataset
from configs import data_config
from stuperml.data import main, MyDataset
import torch

from pathlib import Path

def test_data_preprocessing(): 
    path = Path(data_config.data_folder)    
    assert path.exists(), f"Data directory was not created at: {path.absolute()}"
    
    dataset = MyDataset(cfg=data_config)
    dataset.preprocess()
    
    pt_files = list(path.glob("*.pt"))
    assert len(pt_files) == 6, f"Preprocessing failed. Found files: {[file.name for file in pt_files]}"
    
    pt_expected_files = data_config.file_names
    pt_files_names = [f.name for f in pt_files]
    assert sorted(pt_expected_files) == sorted(pt_files_names), "Filenames does not match with as expected"
    
    pt_file_dict = {
        file.name: file.stat().st_size 
        for file in pt_files 
        if file.stat().st_size < 1e-1
    }    
    assert len(pt_file_dict) == 0, f"File is empty{pt_file_dict}"
    
def test_data_load():
    dataset = MyDataset(cfg=data_config)
    dataset.preprocess()
    
    train_set, validation_set, test_set = MyDataset(cfg=data_config).load_data()
    assert isinstance(train_set, torch.utils.data.TensorDataset)
    
    assert len(train_set) > 0, "train_set is empty"
    assert len(validation_set) > 0, "validation_set is empty"
    assert len(test_set) > 0, "test_set is empty"

    features, targets = train_set.tensors
    assert features.shape[1] > 0
    assert features.shape[0] == targets.shape[0]
    
def test_data_main():
    try: main()
    except Exception as e: 
        assert False, f"main() failed: {e}"
        


if __name__ == '__main__':
    # test_data_preprocessing()
    test_data_load()
    
    
        
