import torch
import torch.nn as nn



class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()

        self.query_conv_x = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_x = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv_x = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma_x = nn.Parameter(torch.zeros(1))

        self.query_conv_y = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_y = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv_y = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma_y = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()


        q1 = self.query_conv_x(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N(W*H) X C
        k1 = self.key_conv_x(x).view(m_batchsize, -1, width * height)  # B X C x N(W*H)
        energy1 = torch.bmm(q1, k1)  # transpose check
        attention1 = self.softmax(energy1)  # BX (N) X (N)
        v1 = self.value_conv_x(x).view(m_batchsize, -1, width * height)  # B X C X N

        q2 = self.query_conv_y(y).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        k2 = self.key_conv_y(y).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy2 = torch.bmm(q2, k2)  # transpose check
        attention2 = self.softmax(energy2)  # BX (N) X (N)
        v2 = self.value_conv_y(y).view(m_batchsize, -1, width * height)  # B X C X N

        out1 = torch.bmm(v1, attention2.permute(0, 2, 1)).view(m_batchsize, C, width, height)
        out2 = torch.bmm(v2, attention1.permute(0, 2, 1)).view(m_batchsize, C, width, height)

        out1 = self.gamma_x * out1 + x
        out2 = self.gamma_y * out2 + y

        return out1, out2

if __name__ == '__main__':
    x = torch.randn(size=(1, 16, 32, 32))
    y = torch.randn(size=(1, 16, 32, 32))
    net = Self_Attn(16)
    out1, out2 = net(x, y)
    print(out1.shape)
    print(out2.shape)
