import torch
import torch.nn as nn

### Model definition ###

class GLSTM(nn.Module):
    def __init__(self, hidden_size=1024, groups=4):
        super(GLSTM, self).__init__()
   
        hidden_size_t = hidden_size // groups
     
        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t//2, 1, batch_first=True, bidirectional=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t//2, 1, batch_first=True, bidirectional=True) for i in range(groups)])
     
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
     
        self.groups = groups
     
    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
    
        out = torch.chunk(out, self.groups, dim=-1)

        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)
    
        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
   
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, grate):
        super(DenseConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, grate, (1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(in_channels+grate, grate, (1,3), padding=(0,1))
        self.conv3 = nn.Conv2d(in_channels+2*grate, grate, (1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(in_channels+3*grate, grate, (1,3), padding=(0,1))
        self.conv5 = GluConv2d(in_channels+4*grate, out_channels, kernel_size, padding=padding, stride=stride)
 
        self.bn1 = nn.BatchNorm2d(grate)
        self.bn2 = nn.BatchNorm2d(grate)
        self.bn3 = nn.BatchNorm2d(grate)
        self.bn4 = nn.BatchNorm2d(grate)
        
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()
  
    def forward(self, x):
        out = x
        out1 = self.elu1(self.bn1(self.conv1(out)))
        out = torch.cat([x, out1], dim=1)
        out2 = self.elu2(self.bn2(self.conv2(out)))
        out = torch.cat([x, out1, out2], dim=1)
        out3 = self.elu3(self.bn3(self.conv3(out)))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out4 = self.elu4(self.bn4(self.conv4(out)))
        out = torch.cat([x, out1, out2, out3, out4], dim=1)
        out5 = self.conv5(out)
    
        out = out5
    
        return out

            
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = DenseConv2d(2, 4, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2 = DenseConv2d(4, 8, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3 = DenseConv2d(8, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4 = DenseConv2d(16, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv5 = DenseConv2d(32, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv6 = DenseConv2d(64, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv7 = DenseConv2d(128, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)
                
        self.glstm = GLSTM(4*256, 4)
        self.fc = nn.Linear(4*256, 1)
        
    def forward(self, x):

        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)

        out = self.glstm(e7)
        out = self.fc(out)
        out = out.permute(0,2,1)

        return out