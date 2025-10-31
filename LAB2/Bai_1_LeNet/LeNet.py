import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2 
        )

        # Layer 2: Pool1
        self.pooling_1 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # Layer 3: Conv2 
        self.conv_2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0  
        )

        # Layer 4: Pool2
        self.pooling_2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # Layer 5: Conv3 
        self.conv_3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            padding=0
        )

        # Layer 6: FC1
        self.fc = nn.Linear(
            in_features=120,
            out_features=84
        )

        # Layer 7: Outp
        self.output = nn.Linear(
            in_features=84,
            out_features=10
        )

    def forward(self, images: torch.Tensor):
        '''
        Input: images shape = (B, 1, 28, 28)
        Output: logits shape = (B, 10)
        '''
        
        x = F.relu(self.conv_1(images))  
        x = self.pooling_1(x)         

        x = F.relu(self.conv_2(x))    
        x = self.pooling_2(x)             

        x = F.relu(self.conv_3(x))        

        x = x.view(x.size(0), -1)     

        x = F.relu(self.fc(x))           
        logits = self.output(x)       
        return logits
