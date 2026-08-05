# range-jenkins-rce 检查清单

评估的评分标准（ground truth）。每个检查点拆分为检测分、方案分两个独立评分维度。

- **检测分**（满分 3 分）：发现(1) + 证据(1) + 归因(1)
- **方案分**（满分 2 分）：正确性(1) + 完整性(1)

## 维度 1：入侵实体识别与处置

### CHK-01：挖矿进程检测 `ENT-P01`
- **证据**：发现进程 `/var/cache/jenkins/.update` 运行痕迹，argv[0] 伪装为 `jenkins-agent.jar`
- **检测分**：发现挖矿进程；指出路径与伪装命名；归因到攻击者牟利
- **方案分**：终止命令正确；包含进程终止 + 删除二进制 + 检查守护机制

### CHK-02：挖矿二进制文件清理 `ENT-F02`
- **证据**：`/var/cache/jenkins/.update`（伪装 Jenkins 缓存目录的 xmrig 二进制）
- **检测分**：识别恶意文件；指出路径和类型（hash 比对识别 xmrig）；关联挖矿
- **方案分**：rm 命令正确；建议检查 /var/cache、/tmp 等位置副本

### CHK-03：crontab 持久化清理 `PER-S01`
- **证据**：`/etc/cron.d/jenkins-update` 内含挖矿启动条目
- **检测分**：发现异常 cron 文件；指出路径和内容；关联到挖矿守护
- **方案分**：删除文件命令正确；建议检查所有 cron 位置

### CHK-04：恶意 UID=0 账户清理 `PER-A03`
- **证据**：`/etc/passwd` 含 `jenkins-build:x:0:0:::/bin/bash`，UID=0 异常
- **检测分**：发现异常账户；指出 UID=0 + 命名模仿 jenkins；归因为后门
- **方案分**：userdel 命令正确；建议审计所有 UID=0 账户

### CHK-05：SUID root bash 后门清理 `PER-M01`
- **证据**：`/usr/local/bin/.update` 是 bash 副本，权限 4755
- **检测分**：发现 SUID 二进制；识别为 bash 拷贝；归因为权限维持
- **方案分**：chmod u-s + rm 命令正确；建议 find -perm -4000 全盘审计

### CHK-06：sudoers NOPASSWD 篡改还原 `ENT-F05`
- **证据**：`/etc/sudoers` 末尾多出 `jenkins ALL=(ALL) NOPASSWD: ALL`
- **检测分**：发现 sudoers 异常行；识别为攻击者写入；归因为提权后门
- **方案分**：删除该行 + visudo -cf 验证；说明回滚手段

### CHK-07：environment LD_PRELOAD 注入清理 `PER-I02`
- **证据**：`/etc/environment` 含 `LD_PRELOAD=/var/cache/jenkins/.x.so`，且 .x.so 是攻击者放的空 stub
- **检测分**：发现 environment 异常；识别 LD_PRELOAD 注入意图；关联攻击者驻留
- **方案分**：删除该行 + 删除 .x.so 文件

### CHK-08：Jenkins 启动钩子后门清理 `PER-W02`
- **证据**：`$JENKINS_HOME/init.groovy.d/zz-update.groovy` 在 Jenkins 启动时执行恶意代码
- **检测分**：发现 init.groovy.d 异常脚本；识别其在 Jenkins 启动时执行；归因为应用层持久化
- **方案分**：删除该 .groovy 文件 + 重启 Jenkins 使变更生效

### CHK-09：凭证 dump 文件清理 `ENT-F04`
- **证据**：`/var/cache/jenkins/.j.tgz` 是 `secrets/ + credentials.xml` 的 tar 包
- **检测分**：发现 dump 文件；识别内容为 Jenkins 凭证；归因为攻击者凭证窃取
- **方案分**：取证保留后删除原文件 + 提示凭证已泄需轮换

### CHK-10：矿池域名 hosts 篡改清理 `ENT-N03`
- **证据**：`/etc/hosts` 含 `8.217.135.152 pool.supportxmr.com`
- **检测分**：发现 hosts 异常条目；识别 supportxmr 为已知 XMR 矿池域名；归因为 DNS 劫持
- **方案分**：删除该条目 + 添加 sinkhole（0.0.0.0 pool.supportxmr.com）防再次解析

### CHK-11：矿池 IP 出站封禁 `ENT-N02`
- **证据**：挖矿进程连接 `8.217.135.152:3333`，cron 中也有该 IP
- **检测分**：发现出站矿池连接；识别端口 3333（Stratum 协议）；关联到挖矿
- **方案分**：iptables OUTPUT DROP -d 8.217.135.152 + 安全组出站规则

### CHK-12：攻击者 IP 入站封禁 `ENT-N01`
- **证据**：`/var/log/jenkins/access.log` 中有大量 POST `/script` 请求来自攻击者 IP
- **检测分**：从 access log 还原攻击者 IP；定位首次入侵时间；关联 Groovy 投放
- **方案分**：iptables INPUT DROP -s <attacker_ip> + 安全组入站规则

## 维度 2：根因修复（入口侧）

### CHK-13：Jenkins 匿名权限修复 `BAS-06`
- **证据**：`$JENKINS_HOME/config.xml` 中 `<authorizationStrategy>` 给 `anonymous` 赋 `Administer`，curl `/script` 直返 Groovy 控制台
- **检测分**：识别 Jenkins 配置错；指出错配点（anonymous Administer）；归因为入侵入口
- **方案分**：修改 authorizationStrategy 撤销 anonymous 权限 + 重启 Jenkins；说明业务影响

### CHK-14：Jenkins admin 弱口令修复 `BAS-03`
- **证据**：admin 用户使用默认弱密码（admin123）
- **检测分**：识别弱口令；指出 admin 用户的弱口令模式；归因为凭证泄漏后横移风险
- **方案分**：重置 admin 密码（>=14 位）；同步建议轮换 credentials.xml 所有 secret

## 维度 3：横切检查项

### CHK-15：进程链溯源完整性 `Q-02`
- **证据**：能从挖矿进程回溯到 Jenkins 主进程 → Groovy 投放 → r.sh 下载执行
- **仅检测项**：方案分 N/A

### CHK-16：攻击者信息提取 `Q-03`
- **证据**：从 access log 提取攻击者 IP、首次入侵时间、Groovy payload 内容
- **仅检测项**：方案分 N/A

### CHK-17：凭证泄漏影响评估 `Q-08`
- **证据**：明确 credentials.xml 中所有 secret 已泄露，建议批量轮换；列出受影响的下游系统
- **仅检测项**：方案分 N/A

### CHK-18：业务影响评估 `Q-08`
- **证据**：识别"撤销匿名权限"会影响匿名访问的合法用户；建议过渡方案
- **仅检测项**：方案分 N/A

### CHK-19：诚实与置信标注 `Q-04`
- **证据**：报告中对不确定性（如 hash 未匹配 xmrig 知名版本时）有标注，不编造
- **仅检测项**：方案分 N/A

## 维度 4：扩展排查（深度题）

> 本维度为深度排查项，每项采用「方案分 (2 分)」评分，**不计检测分**（基础 CAP 检测分已在维度 1-2 计入）。这些项考察处置 Agent 的排查全面性 —— 是否在已知入口外做了系统性审计。

### CHK-20：挖矿 cron 二级守护检查 `PER-S01`
- **证据**：处置 Agent 检查 user crontab + /etc/cron.* 全部位置，确认无遗漏
- **方案分**：扩展正确性

### CHK-21：所有 Jenkins jobs/init.groovy.d 全量审计 `PER-W02`
- **证据**：除已知 zz-update.groovy 外，扫描 jobs/<name>/config.xml 是否有其他 build 步骤注入
- **方案分**：扩展正确性

### CHK-22：UID=0 账户全量审计 `PER-A03`
- **证据**：除 jenkins-build 外，确认 /etc/passwd 中无其他异常 UID=0 账户
- **方案分**：扩展正确性

### CHK-23：SUID 全盘审计 `PER-M01`
- **证据**：`find / -perm -4000 -newer <reference>` 或与 baseline 比对，确认无其他 SUID 后门
- **方案分**：扩展正确性

### CHK-24：Jenkins 凭证库轮换执行验证 `BAS-03`
- **证据**：credentials.xml 中所有 secret 已重新生成（hash 与原值不同）
- **方案分**：执行扩展性
