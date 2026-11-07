## NASA-SR-005 (gold partial -> agent gap; reasoning failure)

**Requirement:** Changes to NASA software requirements shall be logged in the issue tracker by the engineer making the change.

**Expected SWE:** SWE-053 (rank: legacy 2, normalized 1) | **Cited:** SWE-053

**Agent reasoning:** NASA SWE-053 mandates the project manager to track and manage changes, not the engineer. The proposed requirement shifts responsibility to the engineer, which contradicts the obligation.

**Gold rationale:** Logging requirement changes provides recording and tracking, but no obligation to manage them. The cited clause does not itself mandate a CCB or a particular approval procedure. (gap type: Missing obligation to manage requirement changes)

---

## NASA-SR-008 (gold gap -> agent partial; reasoning failure)

**Requirement:** NASA software developers shall write clean, maintainable code following good engineering judgment.

**Expected SWE:** SWE-061 (rank: legacy 29, normalized 8) | **Cited:** SWE-061

**Agent reasoning:** SWE-061 mandates project managers to define and adhere to coding methods, standards, and criteria, which includes clean and maintainable code. The proposed requirement lacks specificity on how 'good engineering judgment' is defined or enforced, missing explicit adherence to formal standards. Thus, it partially meets the obligation.

**Gold rationale:** Clean code and engineering judgment do not select, define, or commit adherence to software coding methods, standards, and criteria. (gap type: No selected/defined coding standard)

---

## NASA-SR-012 (gold meets -> agent partial; reasoning failure)

**Requirement:** The NASA Project Manager shall test the flight and ground software against its baselined software requirements, using documented test procedures and pass/fail criteria traceable to each requirement.

**Expected SWE:** SWE-066 (rank: legacy 4, normalized 3) | **Cited:** SWE-066, SWE-192

**Agent reasoning:** SWE-066 mandates testing against software requirements, which is covered. However, the proposed requirement specifies testing against 'baselined' requirements and includes 'pass/fail criteria traceable to each requirement', which is not explicitly required in SWE-066 or SWE-192. SWE-192 requires verification for hazardous events, but does not mandate traceability of pass/fail criteria. Thus, traceability is missing.

**Gold rationale:** Requires software testing against requirements. (gap type: nan)

---

## NASA-SR-014 (gold partial -> agent gap; reasoning failure)

**Requirement:** For safety-critical NASA flight software components, the Project Manager shall achieve high structural test coverage as demonstrated by the test team.

**Expected SWE:** SWE-219 (rank: legacy 4, normalized 4) | **Cited:** SWE-219, SWE-192, SWE-066, SWE-189, SWE-190

**Agent reasoning:** SWE-219 mandates 100% MC/DC coverage for safety-critical components, not just high test coverage. The proposed requirement fails to specify MC/DC or any formal coverage criterion, missing a core obligation.

**Gold rationale:** Positively requires structural test coverage for safety-critical components but omits 100 percent and MC/DC. The source note allows deviations through technical authority review and waiver; it does not make an unspecified alternate standard automatically sufficient. (gap type: Coverage target not quantified to the required standard)

---

## NASA-SR-017 (gold partial -> agent meets; reasoning failure)

**Requirement:** The NASA Project Manager shall test the software cybersecurity mitigations identified in the Cybersecurity Risk Register.

**Expected SWE:** SWE-159 (rank: legacy 1, normalized 1) | **Cited:** SWE-159

**Agent reasoning:** SWE-159 mandates the project manager to test software cybersecurity mitigations identified from security vulnerabilities and weaknesses analysis, which aligns with the proposed requirement. The requirement specifies testing of mitigations from the Cybersecurity Risk Register, consistent with SWE-159's obligation. No gaps or contradictions exist.

**Gold rationale:** Requires testing of cybersecurity mitigations but omits recorded results and does not establish vulnerability/weakness-analysis provenance of the register. (gap type: Test results not recorded)

---

## NASA-SR-018 (gold meets -> agent partial; reasoning failure)

**Requirement:** The NASA Project Manager shall identify, record, and implement secure coding practices (e.g., input validation, least privilege, secure error handling) in the Secure Coding Standard, and shall verify adherence during code review.

**Expected SWE:** SWE-207 (rank: legacy 1, normalized 1) | **Cited:** SWE-207, SWE-185

**Agent reasoning:** SWE-207 mandates identification, recording, and implementation of secure coding practices, which is covered. However, SWE-185 requires verification via static analysis tools, not just code review, creating a gap in verification method. The proposed requirement omits tool-based verification.

**Gold rationale:** Requires identification, recording, and implementation of secure coding practices. (gap type: nan)

---

## NASA-SR-022 (gold partial -> agent gap; reasoning failure)

**Requirement:** NASA software management and technical measurements shall be collected by each subsystem lead and stored in a shared spreadsheet.

**Expected SWE:** SWE-090 (rank: legacy 17, normalized 5) | **Cited:** SWE-090

**Agent reasoning:** SWE-090 mandates the project manager to establish, record, maintain, report, and utilize software management and technical measurements, not subsystem leads using a shared spreadsheet. The proposed requirement shifts responsibility to subsystem leads and uses a spreadsheet, which is not specified or authorized in NASA NPR 7150.2D.

**Gold rationale:** Collection and spreadsheet storage cover some measurement activity, but no obligation to maintain, report, or utilize the measurements is present. (gap type: No reporting, maintenance, or utilization of measurements)

---

## NASA-SR-024 (gold partial -> agent gap; reasoning failure)

**Requirement:** NASA software non-conformances shall be assigned by the reporting engineer to severity level 'important' or 'not important'.

**Expected SWE:** SWE-202 (rank: legacy 2, normalized 1) | **Cited:** SWE-202

**Agent reasoning:** SWE-202 mandates the project manager to define and implement software severity levels for all non-conformances, not just assign them to 'important' or 'not important'. The proposed requirement fails to specify that severity levels must be defined and implemented by the project manager, only assigning levels. This omits the core obligation of defining levels, creating a gap.

**Gold rationale:** Implements categories but leaves their severity criteria undefined; the clause requires clear levels. (gap type: Undefined severity criteria)

---

## NASA-SR-027 (gold meets -> agent partial; reasoning failure)

**Requirement:** The NASA Project Manager shall collect the number of changes to the spacecraft navigation software requirements each month, track those changes in a requirements log, and report the results to the project team at monthly reviews.

**Expected SWE:** SWE-200 (rank: legacy 6, normalized 4) | **Cited:** SWE-200, SWE-053, SWE-080

**Agent reasoning:** SWE-200 mandates collection and reporting of software requirements volatility metrics, which includes tracking changes. The proposed requirement specifies monthly collection and reporting of change counts, but does not explicitly address volatility metrics or traceability. SWE-053 and SWE-080 cover change tracking, but lack the monthly reporting detail. Missing metric scope and volatility focus creates a partial coverage gap.

**Gold rationale:** Requires collecting, tracking, and reporting the count of requirement changes, a volatility metric. (gap type: nan)

---

## NASA-SR-033 (gold partial -> agent meets; reasoning failure)

**Requirement:** The NASA Project Manager shall hold monthly meetings with project stakeholders to review the spacecraft navigation software’s schedule, development activities, performance metrics, and assessment results.

**Expected SWE:** SWE-018 (rank: legacy 2, normalized 1) | **Cited:** SWE-018

**Agent reasoning:** SWE-018 mandates regular reviews of schedule, status, performance metrics, and assessment results with stakeholders; the proposed requirement aligns by specifying monthly meetings covering these elements. No gaps or contradictions exist.

**Gold rationale:** Requires regular stakeholder reviews of schedule, activities, metrics, and assessments but omits tracking issues to resolution. (gap type: Missing issue tracking to resolution)

---

## NASA-SR-034 (gold partial -> agent meets; reasoning failure)

**Requirement:** The NASA Project Manager shall conduct regression testing after changes to the spacecraft navigation software to confirm that previously working features still function correctly.

**Expected SWE:** SWE-191 (rank: legacy 1, normalized 1) | **Cited:** SWE-191

**Agent reasoning:** SWE-191 mandates regression testing to ensure defects are not introduced into previously integrated software, which aligns with the proposed requirement. The requirement specifies testing after changes to navigation software, consistent with the obligation to verify previously working features. No gaps or contradictions exist.

**Gold rationale:** Requires regression testing for preserved functionality but omits planning and a demonstration that changes did not introduce security vulnerabilities. (gap type: Missing security vulnerability testing)

---

## NASA-SR-037 (gold partial -> agent meets; reasoning failure)

**Requirement:** The NASA Project Manager shall record problems found in the spacecraft navigation software in a defect log when they are first reported.

**Expected SWE:** SWE-201 (rank: legacy 2, normalized 1) | **Cited:** SWE-201

**Agent reasoning:** SWE-201 mandates the project manager to track and maintain software non-conformances including defects, which includes recording defects in software when first reported. The proposed requirement is a direct implementation of this obligation.

**Gold rationale:** Initial defect logging supplies records but not ongoing tracking and maintenance or explicit applicable tools/ground-software scope. (gap type: Missing ongoing maintenance of problem records)

---

## NASA-SR-039 (gold partial -> agent meets; reasoning failure)

**Requirement:** The NASA Project Manager shall record and maintain measurements of the mission-planning software’s performance throughout development.

**Expected SWE:** SWE-090 (rank: legacy 4, normalized 2) | **Cited:** SWE-090

**Agent reasoning:** SWE-090 mandates the project manager to record, maintain, report, and utilize software management and technical measurements. The proposed requirement aligns with this by specifying recording and maintenance of performance measurements during development. No gaps or contradictions exist; the requirement is a direct fulfillment of the obligation.

**Gold rationale:** Requires recording and maintaining technical performance measurements but omits management measurements, establishment, reporting, and use. (gap type: Missing measurement reporting and use)

---

