import torch
from torch import nn
from torch.nn import functional as F

class attention_block(nn.Module):
    def __init__(self):
        super(attention_block, self).__init__()
        self.name = 'attention_block'
        # Global Feature and Global Score
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(3, 2)
        )
        self.conv1_1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.conv2_bn = nn.BatchNorm2d(8)

        self.branch1x1 = nn.Conv2d(8, 4, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(8, 8, kernel_size=1)
        self.branch5x5_1_2 = nn.Conv2d(8, 4, kernel_size=5, padding=2)
        self.branch3x3db1_1 = nn.Conv2d(8, 8, kernel_size=1)
        self.branch3x3db1_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.branch3x3db1_3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(8, 4, kernel_size=1)

        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.conv2_bn1 = nn.BatchNorm2d(8)

        self.branch1x1_2 = nn.Conv2d(8, 4, kernel_size=1)
        self.branch5x5_1_2_1 = nn.Conv2d(8, 8, kernel_size=1)
        self.branch5x5_1_2_2 = nn.Conv2d(8, 4, kernel_size=5, padding=2)
        self.branch3x3db1_1_2 = nn.Conv2d(8, 8, kernel_size=1)
        self.branch3x3db1_2_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.branch3x3db1_3_2 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.branch_pool_2 = nn.Conv2d(8, 4, kernel_size=1)

        self.conv3_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)

        self.conv2_bn2 = nn.BatchNorm2d(8)

        self.branch1x1_3 = nn.Conv2d(8, 4, kernel_size=1)
        self.branch5x5_3_1 = nn.Conv2d(8, 8, kernel_size=1)
        self.branch5x5_3_2 = nn.Conv2d(8, 4, kernel_size=5, padding=2)
        self.branch3x3db1_3_3_1 = nn.Conv2d(8, 8, kernel_size=1)
        self.branch3x3db1_3_3_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.branch3x3db1_3_3_3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.branch_pool_3 = nn.Conv2d(8, 4, kernel_size=1)

        # self.conv4_1 = nn.Conv2d(88, 256, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # self.conv2_bn3 = nn.BatchNorm2d(88)

        self.Global_1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.Global_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.Global_features = nn.Conv2d(8, 4, kernel_size=1, padding=1)
        self.Global_Score = nn.Conv2d(4, 2, kernel_size=1, padding=1)

        # local features and local score************************************************************************************************* # edit upto hear
        self.local_conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)

        # residual_1
        self.residual_1 = nn.Conv2d(4, 8, kernel_size=1)
        self.residual_11 = nn.Conv2d(8, 8, kernel_size=3)
        self.residual_111 = nn.Conv2d(8, 16, kernel_size=1)

        self.local_conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=1)

        # residual_2
        self.residual_2 = nn.Conv2d(16, 16, kernel_size=1)
        self.residual_21 = nn.Conv2d(16, 16, kernel_size=3)
        self.residual_22 = nn.Conv2d(16, 16, kernel_size=1)

        self.local_conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # residual_3
        self.residual_3 = nn.Conv2d(16, 8, kernel_size=1)
        self.residual_31 = nn.Conv2d(8, 8, kernel_size=3)
        self.residual_32 = nn.Conv2d(8, 8, kernel_size=1)

        self.local_conv4 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        #  residual_4
        self.residual_4 = nn.Conv2d(8, 8, kernel_size=1)
        self.residual_41 = nn.Conv2d(8, 8, kernel_size=3)
        self.residual_42 = nn.Conv2d(8, 16, kernel_size=1)

        self.local_conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.local_conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        self.local_features = nn.Conv2d(16, 4, kernel_size=5, padding=2)
        self.local_score = nn.Conv2d(4, 2, kernel_size=7, padding=3)
        ############################################################################

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 1)
        self.pool1 = nn.MaxPool2d(3, 2)
        ##############################################################################
        # main1
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # fully connected
        # self.fc1 = nn.Linear(198*348* 64, 64) # 400 * 700
        # self.fc2 = nn.Linear(64,42)
        # self.fc3 = nn.Linear(42,2)
        ##############################################################################

    def forward(self, x):  # Forward------------------------------------------------

        out1 = self.pool1(F.relu(self.conv2_bn(self.conv1_3(F.relu(self.conv1_2(F.relu(self.conv1_1(x))))))))

        branch1x1 = self.branch1x1(out1)
        branch5x5 = self.branch5x5_1_2(self.branch5x5_1(out1))
        # branch3x3db1 = self.branch3x3db1_3(self.branch3x3db1_2(self.branch3x3db1_1( out1)))
        # branch_pool = self.branch_pool(F.avg_pool2d( out1, kernel_size=3, stride=1, padding=1))
        outputs = torch.cat([branch1x1, branch5x5], 1)

        out2 = self.pool1(F.relu(self.conv2_bn1(self.conv2_2(F.relu(self.conv2_1(outputs))))))

        branch1x1_2 = self.branch1x1_2(out2)
        branch5x5_2 = self.branch5x5_1_2_2(self.branch5x5_1_2_1(out2))
        # branch3x3db1_2 = self.branch3x3db1_3_2(self.branch3x3db1_2_2(self.branch3x3db1_1_2( out2)))
        # branch_pool_2= self.branch_pool_2(F.avg_pool2d( out2, kernel_size=3, stride=1, padding=1))
        outputs = torch.cat([branch1x1_2, branch5x5_2], 1)

        out3 = self.pool1(F.relu(self.conv2_bn2(self.conv3_2(F.relu(self.conv3_1(outputs))))))
        branch1x1_3 = self.branch1x1_3(out3)
        branch5x5_3 = self.branch5x5_3_2(self.branch5x5_3_1(out3))
        # branch3x3db1_3_3 = self.branch3x3db1_3_3_3(self.branch3x3db1_3_3_2(self.branch3x3db1_3_3_1( out3)))
        # branch_pool_3 = self.branch_pool_3(F.avg_pool2d( out3, kernel_size=3, stride=1, padding=1))
        outputs = torch.cat([branch1x1_3, branch5x5_3, ], 1)
        # print(outputs.shape)
        Global_Score = self.Global_Score(self.Global_features(F.relu(self.Global_2(F.relu(self.Global_1(outputs))))))

        # # # Local features ################################################################################################
        local_out1 = self.pool1(F.relu(self.local_conv1(x)))

        R1 = self.residual_1(local_out1)
        R12 = self.residual_11(R1)
        R13 = self.residual_111(R12)

        local_out2 = self.pool1(F.relu(self.local_conv2(R13)))
        R2 = self.residual_2(local_out2)
        R22 = self.residual_21(R2)
        R23 = self.residual_22(R22)

        local_out3 = self.pool1(F.relu(self.local_conv3(R23)))
        R3 = self.residual_3(local_out3)
        R32 = self.residual_31(R3)
        R33 = self.residual_32(R32)

        local_out4 = self.pool1(F.relu(self.local_conv4(R33)))
        R4 = self.residual_4(local_out4)
        R42 = self.residual_41(R4)
        R43 = self.residual_42(R42)

        local_out5 = self.pool1(F.relu(self.local_conv5(R43)))
        # local_out6 = self.pool(F.relu(self.local_conv6(local_out5)))

        local_score = self.local_score(self.local_features(local_out5))

        Local_score = F.interpolate(local_score, size=(Global_Score.data[0].shape[1], Global_Score.data[0].shape[2]),
                                    mode='bilinear', align_corners=False)
        Score = Local_score + Global_Score

        # Attentiom_map = F.softmax(Score, dim=1)
        # print(Attentiom_map.shape)
        ##############################################################################################################
        # main_out = self.pool1(F.relu(self.conv(x)))
        # print(main_out.shape, Attentiom_map.shape, Score.shape)
        ##############################################################################################################
        # Attentiom_map = Attentiom_map.repeat(3, 64, 1, 1)
        # Attentiom_map = F.interpolate(Attentiom_map, size=(main_out.data[0].shape[2], main_out.data[0].shape[2]), mode='bilinear', align_corners=False)

        global_avg = torch.mean(Score, dim=(2, 3))
        # if main_out.shape[0] == 2:
        #   Attentiom_map= torch.reshape(Attentiom_map,[-1])

        # out1 = torch.matmul(main_out, Attentiom_map)

        # max_pool1 = self.pool(torch.add(main_out, out1))
        # print(max_pool1.shape)
        # final_out = torch.tanh(F.dropout( self.fc3(F.dropout(self.fc2(F.dropout(self.fc1(max_pool1.view(max_pool1.size(0),-1))))))))
        # return final_out
        return global_avg
