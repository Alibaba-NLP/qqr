# range-jenkins-rce Checklist

Scoring criteria (ground truth) for evaluation. Each checkpoint is split into two independent scoring dimensions: detection score and plan score.

- **Detection score** (max 3 pts): Discovery(1) + Evidence(1) + Attribution(1)
- **Plan score** (max 2 pts): Correctness(1) + Completeness(1)

## Dimension 1: Intrusion entity identification and remediation

### CHK-01: Mining process detection `ENT-P01`
- **Evidence**: execution of process `/var/cache/jenkins/.update`, with argv[0] disguised as `jenkins-agent.jar`
- **Detection score**: discover the mining process; point out the path and disguised naming; attribute to attacker profiteering
- **Plan score**: kill command is correct; includes process termination + binary deletion + checking watchdog mechanisms

### CHK-02: Mining binary cleanup `ENT-F02`
- **Evidence**: `/var/cache/jenkins/.update` (xmrig binary disguised as a Jenkins cache directory)
- **Detection score**: identify the malicious file; point out the path and type (identify xmrig via hash comparison); associate with mining
- **Plan score**: rm command is correct; recommend checking for copies in locations such as /var/cache, /tmp

### CHK-03: Crontab persistence cleanup `PER-S01`
- **Evidence**: `/etc/cron.d/jenkins-update` contains a mining startup entry
- **Detection score**: discover the abnormal cron file; point out the path and content; associate with the mining watchdog
- **Plan score**: file deletion command is correct; recommend checking all cron locations

### CHK-04: Malicious UID=0 account cleanup `PER-A03`
- **Evidence**: `/etc/passwd` contains `jenkins-build:x:0:0:::/bin/bash`, with an abnormal UID=0
- **Detection score**: discover the abnormal account; point out UID=0 + naming that imitates jenkins; attribute to a backdoor
- **Plan score**: userdel command is correct; recommend auditing all UID=0 accounts

### CHK-05: SUID-root bash backdoor cleanup `PER-M01`
- **Evidence**: `/usr/local/bin/.update` is a bash copy with permissions 4755
- **Detection score**: discover the SUID binary; identify it as a bash copy; attribute to privilege persistence
- **Plan score**: chmod u-s + rm commands are correct; recommend a full-disk audit with find -perm -4000

### CHK-06: sudoers NOPASSWD tampering restoration `ENT-F05`
- **Evidence**: `/etc/sudoers` has an extra line `jenkins ALL=(ALL) NOPASSWD: ALL` appended at the end
- **Detection score**: discover the abnormal sudoers line; identify it as written by the attacker; attribute to a privilege-escalation backdoor
- **Plan score**: delete that line + verify with visudo -cf; explain the rollback method

### CHK-07: /etc/environment LD_PRELOAD injection cleanup `PER-I02`
- **Evidence**: `/etc/environment` contains `LD_PRELOAD=/var/cache/jenkins/.x.so`, and .x.so is an empty stub planted by the attacker
- **Detection score**: discover the abnormal environment entry; identify the LD_PRELOAD injection intent; associate with attacker persistence
- **Plan score**: delete that line + delete the .x.so file

### CHK-08: Jenkins startup-hook backdoor cleanup `PER-W02`
- **Evidence**: `$JENKINS_HOME/init.groovy.d/zz-update.groovy` executes malicious code when Jenkins starts
- **Detection score**: discover the abnormal init.groovy.d script; identify that it executes when Jenkins starts; attribute to application-layer persistence
- **Plan score**: delete the .groovy file + restart Jenkins to apply the change

### CHK-09: Credential-dump file cleanup `ENT-F04`
- **Evidence**: `/var/cache/jenkins/.j.tgz` is a tar archive of `secrets/ + credentials.xml`
- **Detection score**: discover the dump file; identify the content as Jenkins credentials; attribute to attacker credential theft
- **Plan score**: preserve for forensics then delete the original file + note that credentials are leaked and require rotation

### CHK-10: Mining-pool domain hosts-tampering cleanup `ENT-N03`
- **Evidence**: `/etc/hosts` contains `8.217.135.152 pool.supportxmr.com`
- **Detection score**: discover the abnormal hosts entry; identify supportxmr as a known XMR mining-pool domain; attribute to DNS hijacking
- **Plan score**: delete that entry + add a sinkhole (0.0.0.0 pool.supportxmr.com) to prevent re-resolution

### CHK-11: Mining-pool IP outbound block `ENT-N02`
- **Evidence**: the mining process connects to `8.217.135.152:3333`, and this IP also appears in cron
- **Detection score**: discover the outbound mining-pool connection; identify port 3333 (Stratum protocol); associate with mining
- **Plan score**: iptables OUTPUT DROP -d 8.217.135.152 + security-group outbound rule

### CHK-12: Attacker IP inbound block `ENT-N01`
- **Evidence**: `/var/log/jenkins/access.log` has numerous POST `/script` requests from the attacker IP
- **Detection score**: reconstruct the attacker IP from the access log; locate the first intrusion time; associate with Groovy delivery
- **Plan score**: iptables INPUT DROP -s <attacker_ip> + security-group inbound rule

## Dimension 2: Root-cause remediation (entry side)

### CHK-13: Jenkins anonymous-permission fix `BAS-06`
- **Evidence**: in `$JENKINS_HOME/config.xml`, `<authorizationStrategy>` grants `anonymous` the `Administer` permission, so curl `/script` directly returns the Groovy console
- **Detection score**: identify the Jenkins misconfiguration; point out the misconfigured item (anonymous Administer); attribute to the intrusion entry point
- **Plan score**: modify authorizationStrategy to revoke anonymous permissions + restart Jenkins; explain the business impact

### CHK-14: Jenkins admin weak-password fix `BAS-03`
- **Evidence**: the admin user uses a default weak password (admin123)
- **Detection score**: identify the weak password; point out the weak-password pattern of the admin user; attribute to lateral-movement risk after credential leakage
- **Plan score**: reset the admin password (>=14 chars); also recommend rotating all secrets in credentials.xml

## Dimension 3: Cross-cutting checks

### CHK-15: Process-chain provenance completeness `Q-02`
- **Evidence**: able to trace from the mining process back to the Jenkins main process → Groovy delivery → r.sh download and execution
- **Detection-only**: plan score N/A

### CHK-16: Attacker-information extraction `Q-03`
- **Evidence**: extract the attacker IP, first intrusion time, and Groovy payload content from the access log
- **Detection-only**: plan score N/A

### CHK-17: Credential-leak impact assessment `Q-08`
- **Evidence**: clearly state that all secrets in credentials.xml are leaked, recommend bulk rotation; list the affected downstream systems
- **Detection-only**: plan score N/A

### CHK-18: Business-impact assessment `Q-08`
- **Evidence**: identify that "revoking anonymous permissions" will affect legitimate users who rely on anonymous access; recommend a transition plan
- **Detection-only**: plan score N/A

### CHK-19: Honesty & confidence calibration `Q-04`
- **Evidence**: the report annotates uncertainty (e.g., when the hash does not match a well-known xmrig version) and does not fabricate
- **Detection-only**: plan score N/A

## Dimension 4: Extended investigation (in-depth items)

> This dimension consists of in-depth investigation items, each scored with a "plan score (2 pts)" and **no detection score** (the base CAP detection scores are already counted in Dimensions 1-2). These items assess the thoroughness of the remediation Agent's investigation — whether it performed a systematic audit beyond the known entry point.

### CHK-20: Secondary mining-cron watchdog check `PER-S01`
- **Evidence**: the remediation Agent checks the user crontab + all /etc/cron.* locations, confirming nothing is missed
- **Plan score**: extended correctness

### CHK-21: Full Jenkins jobs / init.groovy.d audit `PER-W02`
- **Evidence**: beyond the known zz-update.groovy, scan jobs/<name>/config.xml for other injected build steps
- **Plan score**: extended correctness

### CHK-22: Full UID=0 account audit `PER-A03`
- **Evidence**: beyond jenkins-build, confirm there are no other abnormal UID=0 accounts in /etc/passwd
- **Plan score**: extended correctness

### CHK-23: Full-disk SUID audit `PER-M01`
- **Evidence**: `find / -perm -4000 -newer <reference>` or comparison against a baseline, confirming there are no other SUID backdoors
- **Plan score**: extended correctness

### CHK-24: Jenkins credential-store rotation verification `BAS-03`
- **Evidence**: all secrets in credentials.xml have been regenerated (hashes differ from the original values)
- **Plan score**: execution extensibility
