from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.log import setLogLevel


class SingleSwitchTopo(Topo):
    def build(self):
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')
        h4 = self.addHost('h4')

        self.addLink(s1, s2, bw=20, delay='0ms', loss=5)
        self.addLink(s1, s3, bw=20, delay='0ms', loss=5)
        self.addLink(s1, h1)
        self.addLink(s1, h4)
        self.addLink(s2, h2)
        self.addLink(s3, h3)


def main():
    setLogLevel('info')
    topology = SingleSwitchTopo()
    network = Mininet(topology, host=CPULimitedHost, link=TCLink)
    network.start()
    network.pingAll()
    h1, h2, h3, h4 = network.get('h1', 'h2', 'h3', 'h4')
    network.iperf((h1, h2))
    network.iperf((h1, h3))
    network.iperf((h1, h4))
    network.stop()


if __name__ == '__main__':
    main()
