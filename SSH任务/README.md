# SSH任务

## 一、简述

通过SSH远程链接实现在两台电脑之间传输文件，并学习SSH相关软件的安装和配置。

## 操作环境

- 操作系统：Ubuntu 20.04 LTS
- SSH软件：OpenSSH
- 电脑1 IP地址：192.168.188.36 用户 bacid
- 电脑2 IP地址：192.168.188.57 用户 taylor

## 二、具体操作

### 1.安装OpenSSH
+ 两台电脑都是Ubuntu 20.04 LTS，没有预装openssh，执行如下命令完成安装：
```
sudo apt update
sudo apt install openssh-server
```


### 2.确认ip并连接两台电脑至同一个局域网
分别在两台电脑的终端上执行如下命令查看本机ip：
```
ifconfig
```
![ip](https://github.com/b-Acid/24-vision-lwh/blob/main/SSH%E4%BB%BB%E5%8A%A1/ip.png?raw=true)


可以看到电脑1的ip是192.168.43.81（这是在宿舍连的校园网得到的ip，不同局域网下ip可能不同，接下来的操作所连的局域网下电脑1的ip是192.168.188.36）  


在两台电脑上分别打开终端，确保两台电脑连接在同一局域网中。这里我使用的是自己的手机热点。

### 3.生成公钥和私钥
执行如下命令生成密钥：
```
ssh-keygen
```
输出如下：在Enter passphrase行输入要设置的密码并再输一次确认。
```
Generating public/private rsa key pair.
Enter file in which to save the key (/home/user/.ssh/id_rsa):
Created directory '/home/user/.ssh'.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/user/.ssh/id_rsa.
Your public key has been saved in /home/user/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:47VkvSjlFhKRgz/6RYdXM2EULtk9TQ65PDWJjYC5Jys user@local
The key's randomart image is:
+---[RSA 2048]----+
|       ...o...X+o|
|      . o+   B=Oo|
|       .....ooo*=|
|        o+ooo.+ .|
|       .SoXo.  . |
|      .E X.+ .   |
|       .+.= .    |
|        .o       |
|                 |
+----[SHA256]-----+
```
这个输出表明，生成的公钥放在了 ~/.ssh/id_rsa.pub，私钥放在了 ~/.ssh/id_rsa。接下来，把公钥发送给远程机器，让远程机器记住本机的公钥。
执行如下命令发送公钥：
```
ssh-copy-id user@remote -p port
```

这里user是对方的用户名，remote是ip。分别在两台电脑上执行上述命令，生成公钥并发送给对方。




### 4.通过SSH在电脑1上登录电脑2

在电脑1上，使用以下命令通过SSH登录电脑2：
```
ssh taylor@192.168.188.57
```
输入刚刚设置密码后即成功登录电脑2。

### 5.创建文件夹和文件

  在电脑1上，登录成功电脑2后可以看到终端前面的标头已经变成taylor@@192.168.188.57  使用以下命令创建文件夹和文件并用vim写入内容：
```
cd Documents
mkdir test
cd test
touch test.txt
vim test.txt
```
### 6.修改SSH配置文件

  在电脑1上，使用以下命令修改SSH配置文件：
```
vim ~/.ssh/config
```
在文件中加以下内容：
```
Host TTT
  Hostname 192.168.188.57
  User taylor
```
其中，TTT为电脑2的别名，taylor为电脑2的用户名。这样就实现了通过别名快速登录电脑2的操作。之后要登录电脑2只需要执行如下命令：
```
ssh TTT
```
就实现了免密登录。
  
###  7.通过SCP命令传输文件

在电脑1上，使用以下命令将本机的testvideo.avi文件拷贝至电脑2：
```
scp  testvideo.avi taylolr@192.168.188.57:/home/Documents/
```
在电脑2上的Documents里已经可以看到刚刚传送过来的名为testvideo.avi的文件。  


在电脑1上，使用以下命令将本机的testvideos文件夹拷贝至电脑2：
```
scp  testvideos taylolr@192.168.188.57:/home/Documents/
```
在电脑2上的Documents里已经可以看到刚刚传送过来的名为videos的文件夹。


