# Device Fingerprinting Phase 1 Test Report
**Comparative Evaluation: InAuth vs Fingerprint vs ThreatMetrix**

---

## Executive Summary

**Test Period**: [Start Date] - [End Date]  
**Test Environment**: Sandbox/Controlled Lab Environment  
**Testers**: [Names/Team]  
**Document Version**: 1.0  
**Date**: [Report Date]

### Purpose
This report presents the findings from Phase 1 sandbox testing of three device fingerprinting solutions: InAuth, Fingerprint, and ThreatMetrix. The evaluation focused on device recognition accuracy, privacy resilience, IP intelligence capabilities, and technical performance.

### Key Findings Summary

| Vendor | Overall Assessment | Key Strengths | Primary Concerns |
|--------|-------------------|---------------|------------------|
| **InAuth** | [Pass/Marginal/Fail] | [Key strengths] | [Key concerns] |
| **Fingerprint** | [Pass/Marginal/Fail] | [Key strengths] | [Key concerns] |
| **ThreatMetrix** | [Pass/Marginal/Fail] | [Key strengths] | [Key concerns] |

### Recommendation
**[VENDOR NAME]** is recommended for production implementation based on [brief justification].

---

## 1. Testing Methodology

### 1.1 Test Approach
Our testing methodology evaluated each vendor across three key dimensions:
- **Device Fingerprinting Accuracy**: Ability to consistently recognize devices across various scenarios
- **Privacy Resilience**: Performance when users employ privacy tools and settings
- **IP Intelligence Quality**: Accuracy of network-based risk signals and geolocation

### 1.2 Test Environment
**Devices Used**:
- Mobile: iPhone 14, iPhone 13, Samsung Galaxy S23, Google Pixel 7
- Desktop: Windows 11 Pro, macOS Ventura, Ubuntu 22.04

**Browsers Tested**:
- Mobile: Safari, Chrome, Firefox, Brave
- Desktop: Chrome, Firefox, Safari, Edge

**Network Conditions**:
- Standard residential broadband
- Mobile carrier networks (Verizon, AT&T)
- Corporate VPN
- Commercial VPN services (NordVPN, ExpressVPN)
- Tor network

### 1.3 Test Scenarios
Each vendor was evaluated across 150+ individual test cases covering:
1. **Tolerance Testing** (50 tests): Normal device/browser changes that should NOT break fingerprinting
2. **Privacy Obfuscation Testing** (60 tests): Privacy tools and settings that challenge fingerprinting
3. **Device Reset Testing** (15 tests): Major changes that SHOULD generate new fingerprints
4. **IP Intelligence Testing** (25 tests): Network detection and geolocation accuracy

### 1.4 Assessment Criteria
- **Pass**: Feature works as expected in >85% of test cases
- **Marginal**: Feature works in 60-85% of test cases or has notable limitations
- **Fail**: Feature works in <60% of test cases or has critical gaps

---

## 2. Device Fingerprinting Performance

### 2.1 Overall Device Recognition Accuracy

| Vendor | Assessment | Key Observations |
|--------|------------|------------------|
| InAuth | [Pass/Marginal/Fail] | [Summary of performance] |
| Fingerprint | [Pass/Marginal/Fail] | [Summary of performance] |
| ThreatMetrix | [Pass/Marginal/Fail] | [Summary of performance] |

### 2.2 Tolerance Testing Results

#### Mobile Platforms - iPhone Safari

| Test Scenario | InAuth | Fingerprint | ThreatMetrix | Notes |
|--------------|--------|-------------|--------------|-------|
| OS updates (iOS 17.5 → 17.6) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Browser updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Settings changes (timezone, language) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Network switching (WiFi ↔ Cellular) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Cache/cookie clearing | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Multiple browser windows/tabs | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| App updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |

**Overall iPhone Safari Performance**:
- **InAuth**: [Pass/Marginal/Fail] - [Detailed explanation of results, failure patterns, and impact]
- **Fingerprint**: [Pass/Marginal/Fail] - [Detailed explanation of results, failure patterns, and impact]
- **ThreatMetrix**: [Pass/Marginal/Fail] - [Detailed explanation of results, failure patterns, and impact]

#### Mobile Platforms - iPhone Chrome

| Test Scenario | InAuth | Fingerprint | ThreatMetrix | Notes |
|--------------|--------|-------------|--------------|-------|
| Private browsing consistency | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Settings changes | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Network switching | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Browser updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Cache/cookie clearing | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |

**Overall iPhone Chrome Performance**:
- **InAuth**: [Pass/Marginal/Fail] - [Detailed explanation]
- **Fingerprint**: [Pass/Marginal/Fail] - [Detailed explanation]
- **ThreatMetrix**: [Pass/Marginal/Fail] - [Detailed explanation]

#### Mobile Platforms - Android Chrome

| Test Scenario | InAuth | Fingerprint | ThreatMetrix | Notes |
|--------------|--------|-------------|--------------|-------|
| OS updates (Android 13 → 14) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Browser updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Network switching | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Settings changes | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Cache/cookie clearing | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Incognito mode consistency | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |

**Overall Android Chrome Performance**:
- **InAuth**: [Pass/Marginal/Fail] - [Detailed explanation]
- **Fingerprint**: [Pass/Marginal/Fail] - [Detailed explanation]
- **ThreatMetrix**: [Pass/Marginal/Fail] - [Detailed explanation]

#### Desktop Platforms - Windows 11 Chrome

| Test Scenario | InAuth | Fingerprint | ThreatMetrix | Notes |
|--------------|--------|-------------|--------------|-------|
| Browser version updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| OS security patches | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Extension installations | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Display resolution changes | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Peripheral changes (monitor, mouse) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Network changes | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |

**Overall Windows Chrome Performance**:
- **InAuth**: [Pass/Marginal/Fail] - [Detailed explanation]
- **Fingerprint**: [Pass/Marginal/Fail] - [Detailed explanation]
- **ThreatMetrix**: [Pass/Marginal/Fail] - [Detailed explanation]

#### Desktop Platforms - macOS Safari

| Test Scenario | InAuth | Fingerprint | ThreatMetrix | Notes |
|--------------|--------|-------------|--------------|-------|
| Safari updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| macOS updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| ITP enabled/disabled | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Private browsing consistency | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Display changes | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |

**Overall macOS Safari Performance**:
- **InAuth**: [Pass/Marginal/Fail] - [Detailed explanation]
- **Fingerprint**: [Pass/Marginal/Fail] - [Detailed explanation]
- **ThreatMetrix**: [Pass/Marginal/Fail] - [Detailed explanation]

#### Desktop Platforms - Firefox (All OS)

| Test Scenario | InAuth | Fingerprint | ThreatMetrix | Notes |
|--------------|--------|-------------|--------------|-------|
| Firefox updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Enhanced tracking protection | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Container tabs | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Private browsing | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| OS updates | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |

**Overall Firefox Performance**:
- **InAuth**: [Pass/Marginal/Fail] - [Detailed explanation]
- **Fingerprint**: [Pass/Marginal/Fail] - [Detailed explanation]
- **ThreatMetrix**: [Pass/Marginal/Fail] - [Detailed explanation]

### 2.3 Tolerance Testing Summary

| Platform/Browser | InAuth | Fingerprint | ThreatMetrix |
|-----------------|--------|-------------|--------------|
| iPhone Safari | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] |
| iPhone Chrome | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] |
| Android Chrome | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] |
| Windows Chrome | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] |
| macOS Safari | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] |
| Firefox (All) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] |
| **Overall Tolerance** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** |

**Key Findings**:
- [Major patterns observed across vendors]
- [Platform-specific challenges]
- [Best and worst performers]

### 2.4 Privacy Obfuscation Testing Results

This section evaluates how well each vendor handles privacy-conscious users and privacy-enhancing technologies.

#### Browser Privacy Features

| Privacy Feature | InAuth | Fingerprint | ThreatMetrix | Impact Assessment |
|----------------|--------|-------------|--------------|-------------------|
| **Private/Incognito Mode** | | | | |
| Same ID in private vs regular | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Critical for fraud detection |
| Consistent ID across private sessions | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Critical for fraud detection |
| **Safari ITP (Intelligent Tracking Prevention)** | | | | |
| ID persistence with ITP active | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - Major iOS challenge |
| Cross-domain tracking with ITP | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| **Firefox Enhanced Tracking Protection** | | | | |
| ID persistence with strict mode | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - Aggressive blocking |
| Container tab consistency | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| **Brave Browser Shields** | | | | |
| ID persistence with shields up | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - Privacy-first browser |
| Fingerprint randomization resistance | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High |
| **Edge InPrivate with Tracking Prevention** | | | | |
| ID persistence in InPrivate | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| Strict tracking prevention | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |

**Detailed Findings**:

**InAuth**:
- Private browsing performance: [Detailed results and failure patterns]
- Safari ITP impact: [Specific issues encountered]
- Firefox protection impact: [Specific issues encountered]
- Overall privacy feature resilience: [Pass/Marginal/Fail]

**Fingerprint**:
- Private browsing performance: [Detailed results and failure patterns]
- Safari ITP impact: [Specific issues encountered]
- Firefox protection impact: [Specific issues encountered]
- Overall privacy feature resilience: [Pass/Marginal/Fail]

**ThreatMetrix**:
- Private browsing performance: [Detailed results and failure patterns]
- Safari ITP impact: [Specific issues encountered]
- Firefox protection impact: [Specific issues encountered]
- Overall privacy feature resilience: [Pass/Marginal/Fail]

#### Network Privacy Tools

| Privacy Tool | InAuth | Fingerprint | ThreatMetrix | False Positive Risk |
|-------------|--------|-------------|--------------|-------------------|
| **Commercial VPNs** | | | | |
| NordVPN detection | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| ExpressVPN detection | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| Surfshark detection | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| ProtonVPN detection | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| **Corporate VPNs** | | | | |
| Cisco AnyConnect | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - legitimate use |
| Palo Alto GlobalProtect | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - legitimate use |
| **Privacy Services** | | | | |
| iCloud Private Relay | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - legitimate use |
| Cloudflare WARP | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| **Ad Blockers & Extensions** | | | | |
| uBlock Origin | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - very common |
| Privacy Badger | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| Ghostery | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |

**False Positive Analysis**:

**InAuth**:
- Legitimate users flagged: [Scenarios and frequency]
- Impact on user experience: [Assessment]
- Mitigation strategies: [Recommendations]

**Fingerprint**:
- Legitimate users flagged: [Scenarios and frequency]
- Impact on user experience: [Assessment]
- Mitigation strategies: [Recommendations]

**ThreatMetrix**:
- Legitimate users flagged: [Scenarios and frequency]
- Impact on user experience: [Assessment]
- Mitigation strategies: [Recommendations]

### 2.5 Device Reset Testing Results

These scenarios should generate NEW device IDs to prevent fraud after device transfers or resets.

| Reset Scenario | InAuth | Fingerprint | ThreatMetrix | Expected Behavior |
|---------------|--------|-------------|--------------|-------------------|
| Factory reset (iOS) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | New ID Required |
| Factory reset (Android) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | New ID Required |
| OS reinstall (Windows) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | New ID Required |
| OS reinstall (macOS) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | New ID Required |
| Browser reinstall | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | New ID Required |
| Browser profile reset | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | New ID Required |
| Hardware replacement | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | New ID Required |
| Device restore from backup | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Context-dependent |

**Overall Device Reset Performance**:
- **InAuth**: [Pass/Marginal/Fail] - [Analysis of proper new ID generation]
- **Fingerprint**: [Pass/Marginal/Fail] - [Analysis of proper new ID generation]
- **ThreatMetrix**: [Pass/Marginal/Fail] - [Analysis of proper new ID generation]

**Critical Findings**:
- [Scenarios where vendors incorrectly maintained IDs after reset]
- [Scenarios where vendors incorrectly generated new IDs]
- [Implications for fraud prevention and user experience]

---

## 3. IP Intelligence & Geolocation Performance

### 3.1 Geolocation Approach Comparison

A fundamental difference exists between vendor approaches to location intelligence:

| Aspect | InAuth | Fingerprint | ThreatMetrix |
|--------|--------|-------------|--------------|
| **Primary Method** | [IP-based / HTML5 / Hybrid] | IP-based geolocation | HTML5 Geolocation API |
| **Precision** | [Range] | City-level (5-50km radius) | GPS-level (meters) |
| **User Consent Required** | [Yes/No] | No | Yes |
| **Consent Success Rate** | [N/A or %] | N/A | __%  |
| **Accuracy Indicator** | [Type] | Radius in kilometers | GPS coordinates |
| **User Friction** | [Level] | None | High (permission prompt) |
| **Privacy Approach** | [Description] | Generalized to public areas | Exact location |
| **Fallback Method** | [Description] | IP geolocation database | IP geolocation |
| **Overall Assessment** | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] |

**ThreatMetrix Geolocation Consent Results**:
- Permission prompt shown: [X] times
- User granted permission: [X] times (_%)
- User denied permission: [X] times (_%)
- User ignored prompt: [X] times (_%)
- **Assessment**: [Pass/Marginal/Fail - with justification based on consent rates]

**Key Insight**: [Analysis of the trade-offs between precision and user friction, and which approach better fits business needs]

### 3.2 IP Geolocation Accuracy Testing

#### Country-Level Accuracy

| Region | InAuth | Fingerprint | ThreatMetrix | Test Count |
|--------|--------|-------------|--------------|------------|
| United States | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | 50 tests |
| Canada | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | 20 tests |
| United Kingdom | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | 20 tests |
| Germany | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | 15 tests |
| Australia | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | 15 tests |
| Asia-Pacific | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | 15 tests |
| **Overall Country Accuracy** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | 135 tests |

**Notes**: Pass = >95% accuracy, Marginal = 85-95%, Fail = <85%

#### City-Level Accuracy

| Region Type | InAuth | Fingerprint | ThreatMetrix | Acceptable Range |
|------------|--------|-------------|--------------|------------------|
| Major US Cities (NYC, LA, Chicago) | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Within 50km |
| Secondary US Cities | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Within 50km |
| Rural US Areas | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Within 100km |
| International Major Cities | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Within 50km |
| International Secondary Cities | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Within 75km |
| **Overall City Accuracy** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | |

**Fingerprint Accuracy Radius Analysis**:
- Average accuracy radius: __km
- Range observed: __km to __km
- Most common radius: __km
- Radius reliability: [How actual location related to stated radius - Pass/Marginal/Fail]

**Detailed Findings**:
- **InAuth**: [Geolocation accuracy patterns, strengths, weaknesses]
- **Fingerprint**: [Geolocation accuracy patterns, strengths, weaknesses]
- **ThreatMetrix**: [Geolocation accuracy patterns, strengths, weaknesses]

### 3.3 VPN & Proxy Detection

Critical for detecting location spoofing and fraud.

#### VPN Detection Performance

| VPN Type | InAuth | Fingerprint | ThreatMetrix | False Positive Risk |
|----------|--------|-------------|--------------|-------------------|
| **Commercial VPNs** | | | | |
| NordVPN | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| ExpressVPN | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| Surfshark | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| ProtonVPN | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| CyberGhost | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| **Corporate VPNs** | | | | |
| Cisco AnyConnect | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - legitimate |
| Palo Alto GlobalProtect | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - legitimate |
| Fortinet FortiClient | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - legitimate |
| **Self-Hosted VPNs** | | | | |
| OpenVPN | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| WireGuard | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| **Privacy Services** | | | | |
| iCloud Private Relay | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High - legitimate |
| **Overall VPN Detection** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | |

**Notes**: Pass = >90% detection rate with <5% false positives

#### Proxy Detection Performance

| Proxy Type | InAuth | Fingerprint | ThreatMetrix | False Positive Risk |
|-----------|--------|-------------|--------------|-------------------|
| Data Center Proxies | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| Residential Proxies | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| Mobile Proxies | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High |
| Rotating Proxies | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Medium |
| SOCKS Proxies | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| Transparent Proxies | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | Low |
| **Overall Proxy Detection** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | |

#### Tor Network Detection

| Test | InAuth | Fingerprint | ThreatMetrix | Notes |
|------|--------|-------------|--------------|-------|
| Tor Browser detection | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Tor exit node identification | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| Tor bridge detection | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | |
| **Overall Tor Detection** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | **[Pass/Marginal/Fail]** | |

#### Detection Methods Analysis

**Fingerprint Detection Signals**:

| Method | Assessment | Notes |
|--------|------------|-------|
| Timezone mismatch | [Pass/Marginal/Fail] | [Effectiveness and limitations] |
| Public VPN database | [Pass/Marginal/Fail] | [Coverage and accuracy] |
| OS mismatch | [Pass/Marginal/Fail] | [Platform-specific performance] |
| Relay detection | [Pass/Marginal/Fail] | [iCloud Private Relay, etc.] |

**InAuth Detection Approach**: [Description of methods and assessment]

**ThreatMetrix Detection Approach**: [Description of methods and assessment]

### 3.4 Additional IP Intelligence

| Feature | InAuth | Fingerprint | ThreatMetrix | Business Value |
|---------|--------|-------------|--------------|----------------|
| **Bot Detection** | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | High |
| **ISP Identification** | [Pass/Marginal/Fail] | [Pass/Marginal/Fail] | [Pass/Marginal/
