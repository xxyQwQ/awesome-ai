import os
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel


class SingleSwitchTopo(Topo):
    def build(self):
        h1 = self.addHost('h1', ip='10.0.0.1')
        s1 = self.addSwitch('s1', ip='10.0.0.3')
        self.addLink(s1, h1, bw=10, loss=0, delay='5ms')


def main():
    os.system('sudo ifconfig enp0s8 192.168.56.101 netmask 255.255.255.0')
    setLogLevel('info')
    topology = SingleSwitchTopo()
    network = Mininet(topology, link=TCLink)
    network.start()
    h1 = network.get('h1')
    h1.cmd('ifconfig h1-eth0 10.0.0.1 netmask 255.0.0.0')
    os.system('sudo ifconfig s1 10.0.0.3/8 up')
    os.system('sudo ovs-vsctl add-br br1')
    os.system('sudo ovs-vsctl add-port br1 enp0s8')
    os.system('sudo ifconfig br1 192.168.56.101/24 up')
    os.system('sudo route add default gw 192.168.56.2')
    os.system('sudo ovs-vsctl add-port s1 vx1 -- set interface vx1 type=vxlan options:remote_ip=192.168.56.102')
    CLI(network)
    network.stop()


if __name__ == '__main__':
    main()
