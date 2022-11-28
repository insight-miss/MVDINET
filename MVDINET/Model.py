import torch
import torch.nn as nn
import functools


def get_norm_layer():
    return functools.partial(nn.InstanceNorm2d, affine=False)


def define_MVDINET(norm='batch', num_classes=6, num_view=5, view_list=None, fea_out=200, fea_com=300, **kwargs):
    norm_layer = get_norm_layer()

    MultiviewNet = MVDINET(num_classes=num_classes, num_view=num_view, view_list=view_list,
                           fea_out=fea_out, fea_com=fea_com, **kwargs)
    MultiviewNet.cuda()
    return MultiviewNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Function_CNN(torch.nn.Module):

    def __init__(self, fea_out=512):
        super(Function_CNN, self).__init__()
        self.FC1 = torch.nn.Sequential(
            torch.nn.Linear(19179, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        ).cuda()

        self.FC2 = torch.nn.Sequential(
            torch.nn.Linear(1024, fea_out),
            torch.nn.BatchNorm1d(fea_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        ).cuda()

    def forward(self, x):
        out = self.FC1(x)
        out = self.FC2(out)
        return out


class DeFine_CNN(torch.nn.Module):

    def __init__(self, dropout=0.5):
        super(DeFine_CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        ).cuda()
        self.dropout1 = torch.nn.Dropout(dropout).cuda()

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        ).cuda()
        self.dropout2 = torch.nn.Dropout(dropout).cuda()

    def forward(self, X):
        X = X.reshape(-1, 1, 1000, 6)
        batch_size = X.shape[0]
        out = self.conv1(X)
        out = self.dropout1(out)

        out = self.conv2(out)

        res = self.dropout2(out)

        res = self.conv2(res)
        res = self.dropout2(res)

        return res.reshape(batch_size, -1)


class Special_FC(torch.nn.Module):
    def __init__(self, fea_out=512):
        super(Special_FC, self).__init__()
        self.fea_out = fea_out

    def forward(self, x):
        batch_size, input_channel = x.shape
        out = torch.nn.Sequential(
            torch.nn.Linear(input_channel, self.fea_out),
            torch.nn.BatchNorm1d(self.fea_out),
            torch.nn.ReLU(inplace=True),
        ).cuda()(x)
        return out


class PSSM_NET(nn.Module):
    def __init__(self, seq_len=1000, filters=128, input_dim=20):
        super(PSSM_NET, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=(16, 20), stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=(24, 20), stride=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=(32, 20), stride=1)

        feature1 = int((seq_len - 16) / 1 + 1)
        feature2 = int((seq_len - 24) / 1 + 1)
        feature3 = int((seq_len - 32) / 1 + 1)

        self.max_pool1 = nn.MaxPool2d(kernel_size=(feature1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(feature2, 1))
        self.max_pool3 = nn.MaxPool2d(kernel_size=(feature3, 1))

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(filters * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, 1000, self.input_dim)

        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.max_pool1(x1)
        x1 = self.flatten(x1)
        x1 = self.bn1(x1)

        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x2 = self.max_pool2(x2)
        x2 = self.flatten(x2)
        x2 = self.bn1(x2)

        x3 = self.conv3(x)
        x3 = self.relu(x3)
        x3 = self.max_pool3(x3)
        x3 = self.flatten(x3)
        x3 = self.bn1(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        return self.fc(x)


class MVDINET(nn.Module):
    def __init__(self, num_classes, num_view, view_list, fea_out, fea_com):
        super(MVDINET, self).__init__()

        self.num_view = num_view
        self.fea_out = fea_out
        self.entity_dim = 128
        self.Property_CNN = DeFine_CNN()
        self.Function_FC = Function_CNN()
        self.Special_FC = Special_FC(fea_out=self.fea_out)
        self.PSSM_Network = PSSM_NET(input_dim=20)

        self.PSSM_global = Global_CNN(input_dim=20, out_dim=self.entity_dim)
        self.Property_global = Global_CNN(input_dim=6, out_dim=self.entity_dim)
        self.Domain_global = nn.Sequential(
            nn.Linear(view_list[1], 2 * fea_out),
            nn.BatchNorm1d(2 * fea_out),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2 * fea_out, fea_out),
            nn.BatchNorm1d(fea_out),
            nn.ReLU(inplace=True)
        ).cuda()

        self.embedding = nn.Linear(fea_out, self.entity_dim)

        "512"
        self.relation_out = RelationBlock_Out()

        self.classifier_out = nn.Sequential(
            nn.Linear(num_view * fea_com, fea_com),
            nn.BatchNorm1d(fea_com),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fea_com, num_classes),
            nn.BatchNorm1d(num_classes)
        ).cuda()

        self.local_out = nn.Sequential(
            nn.Linear(fea_com, num_classes),
            nn.BatchNorm1d(num_classes)
        ).cuda()

        self.global_out = nn.Sequential(
            nn.Linear((num_view - 1) * fea_com, fea_com),
            nn.BatchNorm1d(fea_com),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fea_com, num_classes),
            nn.BatchNorm1d(num_classes)
        ).cuda()

    def forward(self, input):

        Special_list = []
        PropertyFeature = self.Property_CNN(input[0])
        FunctionFeature = self.Function_FC(input[1])
        PSSMFeature = self.PSSM_Network(input[2])

        Special_list.append(self.Special_FC(PropertyFeature))
        Special_list.append(self.Special_FC(FunctionFeature))
        Special_list.append(self.Special_FC(PSSMFeature))

        Fea_list = []

        Fea_list.append(self.Property_global(input[0]))
        Fea_list.append(self.embedding(self.Domain_global(input[1])))
        Fea_list.append(self.PSSM_global(input[2]))

        # Deep interactive information
        Relation_fea = self.relation_out(Fea_list)

        Fea_Relation_list = []
        local_list = []
        for k in range(len(Special_list)):
            final_fea = self.local_out(Special_list[k])
            local_list.append(final_fea)
            Fea_Relation_list.append(final_fea)

        global_list = []
        for k in range(len(Fea_list)):
            final_fea = self.global_out(Relation_fea[k])
            global_list.append(final_fea)
            Fea_Relation_list.append(final_fea)

        return Fea_Relation_list

