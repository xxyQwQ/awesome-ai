/*
 * Copyright (c) 2023 Institute of Parallel And Distributed Systems (IPADS), Shanghai Jiao Tong University (SJTU)
 * Licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *     http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "srvmgr.h"
#include "proc_node.h"

void start_daemon_service(void)
{
#ifdef CHCORE_DAEMON_NETCP
        char *args[1];
        args[0] = "/network-cp.service";

        procmgr_launch_process(
                1, args, "network-cp.service", true, INIT_BADGE, NULL, COMMON_APP);

#if (defined CHCORE_ARCH_X86_64) && (defined CHCORE_KERNEL_ENABLE_QEMU_VIRTIO_NET)
        args[0] = "/virtio-net.bin";
        procmgr_launch_process(
                1, args, "virtio-net.bin", true, INIT_BADGE, NULL, SYSTEM_DRIVER);
#endif

#if (defined CHCORE_PLAT_LEON3) && !(defined CHCORE_QEMU)
        args[0] = "/greth.bin";
        procmgr_launch_process(1, args, "greth.bin", true, INIT_BADGE, NULL, SYSTEM_DRIVER);
#endif
#endif
}
