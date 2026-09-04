# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Vortex, please email <security@vortex.dev>.
Please do not report security vulnerabilities through public GitHub issues, discussions, or pull
requests.

To help us triage quickly, include where possible:

- the affected crate, binding, or component and the version or commit you tested against;
- a description of the issue and its impact;
- steps to reproduce, a proof of concept, or a minimal failing input (for example, a crafted
  Vortex file);
- whether the issue is publicly known or being actively exploited.

If you believe the vulnerability is being actively exploited, or you are reporting a severe
incident, follow the [emergency process](#emergency-process) below instead.

## Response Targets

These are targets the maintainers work to, not contractual commitments. Vortex is maintained by
a community of volunteers and employees of several companies.

| Report type | Acknowledgement | Escalation to the CRA steward |
| ----------- | --------------- | ----------------------------- |
| Regular vulnerability report | Within 3 business days | Not required |
| Emergency (`URGENT`) report | Within 24 hours | Immediately, and no later than 24 hours after the maintainers become aware |

## Regular Process

This process applies to vulnerabilities that are not known to be actively exploited.

1. **Report.** The reporter emails <security@vortex.dev> with the details listed above.
2. **Acknowledge.** A maintainer replies to confirm receipt, usually within 3 business days.
3. **Triage.** The maintainers reproduce the issue, assess its severity and affected
   versions, and decide whether it is a security vulnerability. If it is not, we explain why and
   may ask the reporter to open a public issue instead.
4. **Fix.** The maintainers develop a fix privately, using a
   [GitHub security advisory](https://github.com/vortex-data/vortex/security/advisories) and its
   temporary private fork where appropriate. We keep the reporter informed of progress and may
   ask them to validate the fix.
5. **Release.** The fix ships in the next Vortex release. For high-severity issues we cut a
   release as soon as the fix is ready rather than waiting for the regular cadence.
6. **Disclose.** Once the fixed release is available we publish the advisory, request a CVE
   through GitHub, and credit the reporter unless they ask to remain anonymous. We ask
   reporters to hold public disclosure until the fix is released, and we aim to release a fix
   within 90 days of the report.

## Emergency Process

This process applies to:

- a vulnerability in Vortex that is being actively exploited; or
- a severe security incident affecting the project, for example a compromise of the release or
  build process, a published crate, wheel, or binary, project signing keys, or maintainer
  credentials.

Steps for reporters:

1. Email <security@vortex.dev> immediately with `URGENT` at the start of the subject line.
2. Include everything you know about the exploitation or compromise, including how you became
   aware of it. Do not wait to prepare a full write-up.

Steps for maintainers, in order:

1. **Acknowledge** the report within 24 hours and confirm whether it meets the emergency
   criteria above.
2. **Notify the CRA steward.** Notify the project's CRA steward contact at the Linux
   Foundation immediately, and in any case within 24 hours of becoming aware. The steward has
   its own regulatory deadlines starting from that moment. Do this in parallel with fixing the
   problem, never instead of fixing it.
3. **Inform the TSC.** Notify the Vortex Technical Steering Committee so that more than one
   maintainer is coordinating the response.
4. **Contain.** Limit further harm before working on a full fix, for example by yanking a
   compromised crate or wheel, revoking exposed credentials or tokens, and pausing release
   workflows.
5. **Fix and release.** Develop and release a fix or mitigation as quickly as possible. Follow
   the regular process for the fix itself, compressed to whatever timeline the situation
   demands.
6. **Disclose.** Publish an advisory as soon as a fix or mitigation is available, including
   guidance for users who cannot upgrade immediately.
7. **Keep the steward updated.** Send the steward any further information it needs for its
   follow-up notifications and final report, including confirmation of when the fix was
   released.

## Supported Versions

Vortex is under active development and has not yet reached a 1.0 release. Security fixes are
released in the next release from the `develop` branch and are not backported to earlier
releases. Please keep your dependency on Vortex up to date.

## Intended Use

Vortex is published as general-purpose open-source software. It is intended for, and is used
in, commercial products and services as well as non-commercial use.

## CRA Stewardship

Vortex is a project of the LF AI & Data Foundation, hosted as Vortex a Series of LF Projects, LLC.
LF AI & Data is supported under the Linux Foundation CRA stewardship framework. The LF AI & Data
Foundation's CRA steward is The Linux Foundation and its policy is available at
<https://www.linuxfoundation.org/security>.

Under the EU Cyber Resilience Act (CRA), The Linux Foundation, as open-source software steward,
is responsible for regulatory reporting of actively exploited vulnerabilities and severe security
incidents to ENISA. The Vortex project does not operate its own CRA reporting function; it
reports to its steward through the emergency process described above.
