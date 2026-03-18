# =============================================================================
# sdn/ddos_controller.py
# Ryu SDN Controller with ML-driven DDoS mitigation
# Run: ryu-manager sdn/ddos_controller.py
# Requires: pip install ryu  (Python 3.8 recommended for Ryu)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
import logging
from collections import defaultdict, deque
from pathlib import Path

try:
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
    from ryu.ofproto import ofproto_v1_3
    from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
    from ryu.lib import hub
    RYU_AVAILABLE = True
except ImportError:
    RYU_AVAILABLE = False
    print("[!] Ryu not available — running in simulation mode")

import numpy as np
from config import SDN, MODELS_DIR

logger = logging.getLogger(__name__)


# ── Flow statistics tracker ───────────────────────────────────────────────
class FlowTracker:
    """Tracks per-IP flow statistics in a sliding window."""
    WINDOW = 60  # seconds

    def __init__(self):
        self.flows: dict = defaultdict(lambda: {
            "packet_count": deque(maxlen=100),
            "byte_count": deque(maxlen=100),
            "timestamps": deque(maxlen=100),
            "syn_count": 0,
            "blocked": False,
            "block_time": None,
        })

    def update(self, src_ip: str, packet_len: int,
               is_syn: bool = False, is_udp: bool = False):
        now = time.time()
        f = self.flows[src_ip]
        f["packet_count"].append(1)
        f["byte_count"].append(packet_len)
        f["timestamps"].append(now)
        if is_syn:
            f["syn_count"] += 1

    def get_features(self, src_ip: str) -> np.ndarray:
        """Extract flow features for ML inference."""
        f = self.flows[src_ip]
        now = time.time()

        # Recent window packets
        recent = [i for i, t in enumerate(f["timestamps"])
                  if now - t < self.WINDOW]

        pkt_count = len(recent)
        byte_sum  = sum(list(f["byte_count"])[-pkt_count:]) if pkt_count else 0
        syn_count = f["syn_count"]

        # Compute inter-arrival times
        ts = list(f["timestamps"])[-pkt_count:]
        iats = [ts[i+1]-ts[i] for i in range(len(ts)-1)] if len(ts) > 1 else [0]

        features = np.array([
            pkt_count,
            byte_sum,
            byte_sum / max(pkt_count, 1),       # avg pkt size
            pkt_count / max(self.WINDOW, 1),     # packets/sec
            byte_sum / max(self.WINDOW, 1),      # bytes/sec
            min(iats), max(iats),
            np.mean(iats), np.std(iats),
            syn_count,
        ], dtype=np.float32)
        return features


# ── Standalone simulation (when Ryu not available) ────────────────────────
class SDNSimulator:
    """
    Simulates an SDN controller for testing without Mininet.
    Generates synthetic traffic and applies mitigation rules.
    """

    def __init__(self):
        self.tracker   = FlowTracker()
        self.blocked   = set()
        self.log_path  = Path("results") / "sdn_simulation.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events    = []

    def simulate_traffic(self, n_packets: int = 1000):
        """Inject simulated attack and benign traffic."""
        import random
        rng = np.random.default_rng(42)
        attack_ips  = [f"10.0.0.{i}" for i in range(1, 6)]
        benign_ips  = [f"192.168.1.{i}" for i in range(1, 11)]

        print(f"\n  Simulating {n_packets} packets...")
        mitigated = 0
        false_pos = 0

        for _ in range(n_packets):
            # 60% attack, 40% benign
            is_attack = rng.random() < 0.6
            src_ip    = rng.choice(attack_ips if is_attack else benign_ips)
            pkt_len   = int(rng.normal(60, 10) if is_attack else rng.normal(500, 200))
            is_syn    = is_attack and rng.random() < 0.8
            is_udp    = is_attack and rng.random() < 0.5

            self.tracker.update(src_ip, max(40, pkt_len), is_syn, is_udp)
            features = self.tracker.get_features(src_ip)
            score    = self._mock_ml_score(features, is_attack)

            if score > SDN["block_threshold"] and src_ip not in self.blocked:
                self.blocked.add(src_ip)
                event = {
                    "time": time.time(),
                    "action": "BLOCK",
                    "src_ip": src_ip,
                    "score": round(float(score), 4),
                    "true_attack": is_attack,
                }
                self.events.append(event)
                mitigated += 1
                if not is_attack:
                    false_pos += 1
                print(f"    [BLOCK] {src_ip}  score={score:.3f}  "
                      f"{'✓ TP' if is_attack else '✗ FP'}")

        mitigation_rate = mitigated / max(len(set(attack_ips)), 1) * 100
        fp_rate = false_pos / max(len(set(benign_ips)), 1) * 100
        print(f"\n  Mitigation rate : {mitigation_rate:.1f}%")
        print(f"  False positive  : {fp_rate:.1f}%")
        print(f"  Blocked IPs     : {len(self.blocked)}")

        with open(self.log_path, 'w') as f:
            json.dump({
                "total_packets": n_packets,
                "blocked_ips": list(self.blocked),
                "events": self.events[-50:],
                "mitigation_rate": round(mitigation_rate, 2),
                "false_positive_rate": round(fp_rate, 2),
            }, f, indent=2)
        print(f"  [✓] Log → {self.log_path}")

    def _mock_ml_score(self, features: np.ndarray, is_attack: bool) -> float:
        """Simple threshold-based score (replace with real model in production)."""
        pkt_rate = features[3]   # packets/sec
        syn_rate = features[9]   # syn_count
        noise    = np.random.normal(0, 0.05)
        if is_attack:
            score = min(1.0, 0.7 + pkt_rate / 1000 + syn_rate / 100 + noise)
        else:
            score = max(0.0, 0.2 - pkt_rate / 5000 + noise)
        return float(score)


# ── Ryu Controller App ────────────────────────────────────────────────────
if RYU_AVAILABLE:
    class DDoSMitigationController(app_manager.RyuApp):
        """
        OpenFlow 1.3 controller with ML-driven DDoS mitigation.
        Installs drop rules for detected attack sources.
        """
        OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mac_to_port = {}
            self.tracker     = FlowTracker()
            self.blocked_ips = set()
            self.events      = []
            self.monitor_thread = hub.spawn(self._monitor)

        @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
        def switch_features_handler(self, ev):
            """Install table-miss flow entry on switch connection."""
            dp     = ev.msg.datapath
            parser = dp.ofproto_parser
            ofp    = dp.ofproto
            match  = parser.OFPMatch()
            actions = [parser.OFPActionOutput(
                ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
            self._add_flow(dp, 0, match, actions)
            self.logger.info(f"Switch {dp.id} connected")

        @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
        def packet_in_handler(self, ev):
            msg      = ev.msg
            datapath = msg.datapath
            pkt      = packet.Packet(msg.data)
            eth_pkt  = pkt.get_protocol(ethernet.ethernet)
            ip_pkt   = pkt.get_protocol(ipv4.ipv4)
            tcp_pkt  = pkt.get_protocol(tcp.tcp)
            udp_pkt  = pkt.get_protocol(udp.udp)

            if ip_pkt is None:
                return

            src_ip = ip_pkt.src
            if src_ip in self.blocked_ips:
                return  # Already blocked

            is_syn = tcp_pkt is not None and (tcp_pkt.bits & 0x02) != 0
            is_udp = udp_pkt is not None
            pkt_len = len(msg.data)

            self.tracker.update(src_ip, pkt_len, is_syn, is_udp)
            features = self.tracker.get_features(src_ip)
            score    = self._predict_attack(features)

            if score > SDN["block_threshold"]:
                self._block_ip(datapath, src_ip, score)

            # Normal L2 forwarding for non-blocked traffic
            self._forward_packet(ev, eth_pkt, datapath)

        def _predict_attack(self, features: np.ndarray) -> float:
            """
            Hook for ML model inference.
            Replace with: model.predict(features.reshape(1,-1))[0][1]
            """
            pkt_rate = features[3]
            syn_rate = features[9]
            score = min(1.0, pkt_rate / 500 + syn_rate / 50)
            return float(score)

        def _block_ip(self, datapath, src_ip: str, score: float):
            """Install a high-priority drop rule for the attack source."""
            parser = datapath.ofproto_parser
            ofp    = datapath.ofproto
            match  = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
            self._add_flow(datapath, priority=100,
                           match=match, actions=[],
                           idle_timeout=SDN["idle_timeout"])
            self.blocked_ips.add(src_ip)
            self.logger.warning(
                f"BLOCKED {src_ip}  score={score:.3f}  "
                f"timeout={SDN['idle_timeout']}s")

        def _add_flow(self, datapath, priority, match, actions,
                      idle_timeout=0, hard_timeout=0):
            parser = datapath.ofproto_parser
            ofp    = datapath.ofproto
            instr  = [parser.OFPInstructionActions(
                ofp.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(
                datapath=datapath, priority=priority,
                match=match, instructions=instr,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout)
            datapath.send_msg(mod)

        def _forward_packet(self, ev, eth_pkt, datapath):
            """Standard L2 learning switch forwarding."""
            msg    = ev.msg
            ofp    = datapath.ofproto
            parser = datapath.ofproto_parser
            in_port = msg.match['in_port']
            dpid    = datapath.id

            self.mac_to_port.setdefault(dpid, {})
            self.mac_to_port[dpid][eth_pkt.src] = in_port
            out_port = self.mac_to_port[dpid].get(
                eth_pkt.dst, ofp.OFPP_FLOOD)

            actions = [parser.OFPActionOutput(out_port)]
            if out_port != ofp.OFPP_FLOOD:
                match = parser.OFPMatch(in_port=in_port,
                                        eth_dst=eth_pkt.dst)
                self._add_flow(datapath, 1, match, actions)

            data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
            out  = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=data)
            datapath.send_msg(out)

        def _monitor(self):
            """Background thread: log blocked IPs every 30s."""
            while True:
                hub.sleep(30)
                self.logger.info(
                    f"Blocked IPs: {len(self.blocked_ips)}  "
                    f"Active flows: {len(self.tracker.flows)}")


if __name__ == "__main__":
    if RYU_AVAILABLE:
        print("Run: ryu-manager sdn/ddos_controller.py")
        print("Then: sudo mn --controller=remote,ip=127.0.0.1,port=6633")
    else:
        print("Running SDN simulation (Ryu not installed)...")
        sim = SDNSimulator()
        sim.simulate_traffic(n_packets=500)
