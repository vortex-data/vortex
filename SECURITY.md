# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Vortex, please email <vuln-report@vortex.dev>.
Please do not report security vulnerabilities through public GitHub issues, discussions, or pull
requests.

To help us triage quickly, include where possible:

- the affected crate, binding, or component and the version or commit you tested against;
- a description of the issue and its impact;
- steps to reproduce, a proof of concept, or a minimal failing input (for example, a crafted
  Vortex file);
- whether the issue is publicly known or being actively exploited.

The Vortex maintainers will acknowledge your report, work with you to understand and fix the
issue, and coordinate public disclosure once a fix is available. We ask that you give us a
reasonable opportunity to release a fix before disclosing the issue publicly.

## Actively Exploited Vulnerabilities and Severe Incidents

If you believe a vulnerability is being actively exploited, or you become aware of a severe
security incident affecting the project (for example, compromise of the release or build
process, a published crate, wheel, or binary, or project credentials), email
<vuln-report@vortex.dev> immediately with `URGENT` at the start of the subject line.

Vortex maintainers who learn of an actively exploited vulnerability or a severe incident must
notify the project's CRA steward contact at the Linux Foundation immediately, in parallel with
fixing the problem and never instead of fixing it.

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
reports to its steward through the process described above.
