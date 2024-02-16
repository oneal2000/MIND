from torch.utils.data import Dataset
import torch
torch.manual_seed(0)

class TrainDataset(Dataset):
    def __init__(self, train_data, args, typ="train"):
        self.all_data = []

        for _, data in enumerate(train_data):
            self.all_data.append({
                "label": data["label"],
                "hd": data["hd"]
            })
        self.halu_num = len([d for d in self.all_data if d["label"]])
        print(f"{typ} data: [0, 1] - [{len(self.all_data) - self.halu_num}, {self.halu_num}]")
            
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        data = self.all_data[idx]  
        return {
            "input": torch.tensor(data["hd"]),
            "y": torch.LongTensor([data["label"]]),
        }

