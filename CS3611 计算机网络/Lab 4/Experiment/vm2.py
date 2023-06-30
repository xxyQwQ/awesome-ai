import os
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel


class SingleSwitchTopo(Topo):
    def build(self):
        h2 = self.addHost('h2', ip='10.0.0.2')
        s2 = self.addSwitch('s2', ip='10.0.0.4')
        self.addLink(s2, h2, bw=10, loss=0, delay='5ms')


def main():
    os.system('sudo ifconfig enp0s8 192.168.56.102 netmask 255.255.255.0')
    setLogLevel('info')
    topology = SingleSwitchTopo()
    network = Mininet(topology, link=TCLink)
    network.start()
    h2 = network.get('h2')
    h2.cmd('ifconfig h2-eth0 10.0.0.2 netmask 255.0.0.0')
    os.system('sudo ifconfig s2 10.0.0.4/8 up')
    os.system('sudo ovs-vsctl add-br br2')
    os.system('sudo ovs-vsctl add-port br2 enp0s8')
    os.system('sudo ifconfig br2 192.168.56.102/24 up')
    os.system('sudo route add default gw 192.168.56.2')
    os.system('sudo ovs-vsctl add-port s2 vx2 -- set interface vx2 type=vxlan options:remote_ip=192.168.56.101')
    CLI(network)
    network.stop()


if __name__ == '__main__':
    main()
