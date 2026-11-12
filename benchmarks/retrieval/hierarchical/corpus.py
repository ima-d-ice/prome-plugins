"""The labeled evaluation corpus of the hierarchical retrieval strategy.

The corpus is a small, hand-written documentation set with the ground truth
attached to every query.  It deliberately contains the situations the
hierarchical strategy is supposed to handle:

* **long technical manuals** -- a document with many sections and long
  sections, so the flat search has to pick a passage out of a large corpus
  and the hierarchy has to narrow it down;
* **multi-section documentation** -- sections that share the same vocabulary,
  so a query can match several sections equally well;
* **research-style documents** -- abstract, method and results sections that
  use the same terms with a different meaning;
* **repeated terminology across sections** -- the same word family appears in
  several sections of a document, and only one of them answers the query;
* **queries that need a precise piece of evidence inside a large parent** --
  the answer is one sentence in the middle of a long section.

The ground truth is expressed on the *text*, not on the chunk ids: a query
names the sentences that answer it.  The evaluation harness resolves those
sentences to chunk ids after chunking, so the labels stay valid when the
chunking configuration changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    """One section of a corpus document."""

    heading: str
    level: int = 1
    paragraphs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    """One document of the corpus."""

    source: str
    title: str
    sections: tuple[Section, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Query:
    """One labeled query.

    Attributes:
        text: the query as a user would type it.
        relevant_sections: the headings of the sections that answer it.
        relevant_evidence: the exact sentences that carry the answer.  A
            sentence is the ground truth for the *child* level.
    """

    text: str
    relevant_sections: tuple[str, ...]
    relevant_evidence: tuple[str, ...]
    notes: str = ""


# ----------------------------------------------------------------------
# The corpus.
# ----------------------------------------------------------------------
_OPERATIONS = Document(
    source="platform-operations.md",
    title="Platform Operations Manual",
    sections=(
        Section(
            "Overview",
            paragraphs=(
                "This manual describes the day-to-day operation of the data "
                "platform, the on-call rotation and the incident process. It is "
                "written for the engineers who run the platform rather than for "
                "the users of the data it holds.",
                "The platform ingests events from the edge collector and stores "
                "them in a partitioned object store. Downstream jobs read those "
                "partitions through the query service, which is the component "
                "most of the remaining sections of this manual talk about.",
                "Every component of the platform exposes a health endpoint and "
                "writes structured logs to the shared log store. The metrics "
                "that the on-call rotation watches are described in the "
                "monitoring chapter.",
            ),
        ),
        Section(
            "Key Rotation",
            level=2,
            paragraphs=(
                "Signing keys are rotated every thirty days by the key rotation "
                "scheduler, which runs at 02:00 UTC. The scheduler writes the "
                "new key version to the key store and then publishes the change "
                "to the discovery endpoint, so that the collectors pick it up "
                "without downtime.",
                "A key that is not rotated within the grace period of seven days "
                "is revoked automatically and the affected collectors are "
                "restarted in a rolling fashion. The restart is deliberately "
                "slow so that the ingest pipeline absorbs the reconnect storm.",
                "Operators can trigger a rotation by hand, but a manual rotation "
                "does not reset the thirty day schedule. The next scheduled "
                "rotation still happens on its usual day.",
                "The rotation writes an audit entry that records which key "
                "version replaced which one. The audit entry is the source of "
                "truth when a collector reports a signature error after a "
                "rotation window.",
            ),
        ),
        Section(
            "Shard Rebalancing",
            level=2,
            paragraphs=(
                "A shard rebalance moves partitions between nodes so that every "
                "node holds roughly the same amount of data. Rebalancing is a "
                "background activity and does not interrupt the ingest "
                "pipeline.",
                "The rebalance scheduler starts a rebalance when the skew of a "
                "node exceeds twenty percent for more than fifteen minutes. The "
                "fifteen minute window keeps a short burst of traffic from "
                "triggering an expensive operation.",
                "During a rebalance the affected partitions are replicated to "
                "their new owner before the old copy is removed. A rebalance "
                "that is interrupted is resumed from the plan that was written "
                "when it started.",
                "The rebalance plan is stored with the partition metadata, so "
                "the operator can inspect which partitions are in flight and "
                "how far the copy has progressed.",
            ),
        ),
        Section(
            "Backup And Restore",
            level=2,
            paragraphs=(
                "Backups are taken every six hours and are kept for thirty days "
                "in a separate bucket. A backup is a full copy of the object "
                "store, not an incremental snapshot, so restoring one does not "
                "depend on the previous backup.",
                "A restore replays the write-ahead log of the restored range and "
                "then re-applies the compacted segments. The replay is the slow "
                "part of a restore and it dominates the estimated duration.",
                "Restoring a single partition is allowed, restoring a whole "
                "table is an operator-only procedure. A partition restore is "
                "served from the nearest replica and therefore takes minutes "
                "rather than hours.",
                "The backup bucket is in a different region than the live "
                "cluster. A restore therefore also transfers the data across "
                "regions, which is accounted for in the estimate.",
            ),
        ),
        Section(
            "Capacity Planning",
            level=2,
            paragraphs=(
                "Capacity planning uses the ingestion forecast of the previous "
                "quarter to size the cluster for the next one. The forecast is "
                "produced by the analytics group and is reviewed every quarter "
                "by the platform team.",
                "A node is added when the projected storage usage exceeds "
                "seventy percent of the usable capacity. Adding a node early "
                "is cheap, adding one during a busy week is not.",
                "The usable capacity is the total capacity minus the reserve "
                "that is kept for the rebalance of a failed node. The reserve "
                "is one node worth of capacity.",
                "CPU and memory are provisioned with the node and are not "
                "planned separately, because the ingest pipeline is limited by "
                "storage throughput in every profile the platform has measured.",
            ),
        ),
        Section(
            "Log Retention",
            level=2,
            paragraphs=(
                "Application logs are retained for fourteen days, audit logs for "
                "one year and access logs for ninety days. The three classes "
                "have different retention because they answer different "
                "questions: debugging, auditing and access review.",
                "Log retention is enforced by a daily compaction job that drops "
                "the expired segments. The job is idempotent and can be run "
                "again without any effect on the segments it already dropped.",
                "A legal hold suspends the retention of a log partition until "
                "the hold is released. The hold is applied before the compaction "
                "job runs, so the partition survives.",
                "The retention periods are configured per environment. The "
                "production values are the ones listed above; the staging "
                "environment keeps everything for seven days.",
            ),
        ),
        Section(
            "On-Call Rotation",
            level=2,
            paragraphs=(
                "The on-call rotation has one primary and one secondary "
                "engineer per week. The handover happens on Monday morning and "
                "the incoming engineer walks through the open incidents.",
                "An incident is declared when a service level objective is at "
                "risk. The incident commander coordinates the response and "
                "writes the timeline during the incident.",
                "Every incident produces a review. The review is scheduled "
                "within five working days and its action items are tracked like "
                "any other engineering task.",
            ),
        ),
    ),
)

_BILLING = Document(
    source="billing-guide.md",
    title="Billing Integration Guide",
    sections=(
        Section(
            "Overview",
            paragraphs=(
                "This guide describes how invoices are produced, issued and "
                "settled by the billing system. It is aimed at the engineers "
                "who integrate the billing system with a product, not at the "
                "customers who receive the invoices.",
                "The billing system consumes the usage records of the platform "
                "and prices them with the rate card of the contract. A usage "
                "record carries a dimension, a quantity and a timestamp; the "
                "dimensions of a contract decide how a record is priced.",
                "Invoices are produced in two passes. The first pass prices the "
                "usage of the period, the second pass applies the credits, the "
                "taxes and the promotions of the contract.",
            ),
        ),
        Section(
            "Invoice Generation",
            level=2,
            paragraphs=(
                "Invoices are issued on the first day of every month and cover "
                "the calendar month that just ended. An invoice for a partial "
                "month is pro-rated by day, so a contract that starts on the "
                "fifteenth is billed for sixteen days in that month.",
                "The invoice number is stable and is reused by a credit note. "
                "A customer therefore always sees one number per billing "
                "period, even when the invoice was corrected.",
                "An invoice is produced once the usage period is closed. Usage "
                "that arrives after the period is closed is billed in the next "
                "invoice and is marked as a late record on the invoice.",
                "Invoices are delivered to the billing address of the account "
                "and are also available through the billing portal. A customer "
                "who changes the billing address does not change the number of "
                "the invoice.",
            ),
        ),
        Section(
            "Usage Rating",
            level=2,
            paragraphs=(
                "Usage is rated per dimension with the rate card that is "
                "attached to the contract. A dimension that is not mentioned in "
                "the rate card is billed at the default rate of the region.",
                "The default rate is the price of the smallest published unit of "
                "the region. It is a fallback, not a discount, and the usage "
                "report shows the fallback explicitly.",
                "Rating is idempotent: rating the same usage record twice "
                "produces the same charge. The rating key is the record id, so "
                "a repeated run of the rating job does not change an invoice "
                "that was already issued.",
                "Usage that cannot be rated, because the contract is missing or "
                "the dimension is unknown, is put in an exception queue. The "
                "queue is reviewed by the billing operations team every working "
                "day.",
            ),
        ),
        Section(
            "Credits And Refunds",
            level=2,
            paragraphs=(
                "A refund is issued within five business days of an approved "
                "credit note. The credit note itself is approved by the "
                "account manager who requested it.",
                "A partial refund keeps the original invoice number and "
                "reduces the amount that was paid. A full refund keeps the "
                "invoice number as well and marks the invoice as settled by "
                "credit.",
                "Credits above the amount that was invoiced are carried over to "
                "the next invoice. They are not refunded, because the customer "
                "has not paid for them yet.",
                "A refund is issued to the original payment method whenever the "
                "payment method still accepts a refund. Otherwise the credit is "
                "paid out by transfer.",
            ),
        ),
        Section(
            "Tax Handling",
            level=2,
            paragraphs=(
                "Taxes are computed per jurisdiction from the address of the "
                "billing entity. The tax rate that applies is the one in force "
                "on the invoice date, not the one of the day the invoice is "
                "issued.",
                "An exempt customer is identified by a valid exemption "
                "certificate that is stored with the account. The certificate "
                "is checked when the account is created and again at the start "
                "of every billing period.",
                "A customer whose certificate expires during a billing period is "
                "taxed for that period and is exempted again afterwards, once "
                "the new certificate is on file.",
                "Taxes are reported per jurisdiction, which means an invoice "
                "can carry several tax lines. The tax lines are shown "
                "separately on the invoice.",
            ),
        ),
    ),
)

_RESEARCH = Document(
    source="retrieval-study.md",
    title="A Study of Chunking Strategies for Retrieval",
    sections=(
        Section(
            "Abstract",
            paragraphs=(
                "We compare three chunking strategies for a retrieval system "
                "over a technical documentation corpus. The strategies are a "
                "flat chunking, a structure aware chunking and a two level "
                "hierarchy.",
                "The study measures whether a coarse-to-fine hierarchy improves "
                "the precision of the retrieved evidence. We report the "
                "recall of the evidence chunks, the rank of the first correct "
                "answer and the number of candidates that had to be scored.",
                "The hierarchy reduced the number of candidate passages by an "
                "order of magnitude on the documents with many sections that "
                "share the same vocabulary. On the rest of the corpus the two "
                "strategies were close.",
            ),
        ),
        Section(
            "Method",
            paragraphs=(
                "Each document is split into a hierarchy of two levels, a "
                "section level and a passage level. The section level follows "
                "the headings of the document, the passage level is produced by "
                "splitting the section into overlapping windows.",
                "Every level is embedded independently with the same encoder. "
                "The embedding of a section is therefore not reused for its "
                "passages, so no level can inherit the representation of the "
                "other.",
                "Queries are answered by selecting the relevant sections first "
                "and the passages inside them second. We compare this with a "
                "flat search over all passages of the corpus.",
                "Both systems use the same embeddings and the same number of "
                "results, so the difference between them is the order in which "
                "the candidates are generated and not the representation of the "
                "corpus.",
                "The evaluation uses a fixed set of queries with manually "
                "labelled evidence sentences. The label of a query is the "
                "sentence that answers it, not the section that contains it, so "
                "that the passage level is measured as well.",
            ),
        ),
        Section(
            "Results",
            paragraphs=(
                "The coarse-to-fine hierarchy reduced the number of candidate "
                "passages that had to be scored by an order of magnitude. The "
                "reduction was strongest on the documents with the largest "
                "number of sections.",
                "On the precision at one the two strategies were close, and the "
                "hierarchy was ahead on the recall at five. The reason is that "
                "the flat search often places a passage of a neighbouring "
                "section first when the two sections share their vocabulary.",
                "The advantage of the hierarchy was larger on documents with "
                "many sections that share the same vocabulary. On documents "
                "with short and distinct sections the flat search was already "
                "precise and the hierarchy added a second search without "
                "changing the answer.",
                "The latency of the hierarchy is the sum of the two searches, "
                "which was roughly twice the latency of the flat search on the "
                "same hardware.",
            ),
        ),
        Section(
            "Threats To Validity",
            paragraphs=(
                "The corpus is a single technical documentation set, so the "
                "result may not generalize to other domains. In particular, a "
                "corpus of narrative text has fewer sections that share their "
                "vocabulary and would show a smaller advantage.",
                "The encoder was not tuned for the hierarchy, which probably "
                "understates the benefit. A model that was trained on "
                "section and passage pairs would rank the sections better.",
                "The labels were written by one annotator, so the agreement "
                "with a second annotator was not measured.",
            ),
        ),
    ),
)

CORPUS: tuple[Document, ...] = (_OPERATIONS, _BILLING, _RESEARCH)


QUERIES: tuple[Query, ...] = (
    Query(
        text="How often are the signing keys rotated and what happens if one "
        "is not?",
        relevant_sections=("Key Rotation",),
        relevant_evidence=(
            "Signing keys are rotated every thirty days by the key rotation "
            "scheduler, which runs at 02:00 UTC.",
            "A key that is not rotated within the grace period of seven days "
            "is revoked automatically and the affected collectors are "
            "restarted in a rolling fashion.",
        ),
        notes="repeated terminology: 'rotated' also appears in the key "
        "rotation section twice",
    ),
    Query(
        text="When does the scheduler start a shard rebalance and how is the "
        "data moved?",
        relevant_sections=("Shard Rebalancing",),
        relevant_evidence=(
            "The rebalance scheduler starts a rebalance when the skew of a "
            "node exceeds twenty percent for more than fifteen minutes.",
            "During a rebalance the affected partitions are replicated to "
            "their new owner before the old copy is removed.",
        ),
        notes="precise evidence inside a large parent section",
    ),
    Query(
        text="How long are backups kept and can a single partition be "
        "restored?",
        relevant_sections=("Backup And Restore",),
        relevant_evidence=(
            "Backups are taken every six hours and are kept for thirty days "
            "in a separate bucket.",
            "Restoring a single partition is allowed, restoring a whole "
            "table is an operator-only procedure.",
        ),
        notes="two answers in the same section",
    ),
    Query(
        text="For how long are the audit logs retained?",
        relevant_sections=("Log Retention",),
        relevant_evidence=(
            "Application logs are retained for fourteen days, audit logs for "
            "one year and access logs for ninety days.",
        ),
        notes="one sentence in the middle of a long section",
    ),
    Query(
        text="What triggers the addition of a node to the cluster?",
        relevant_sections=("Capacity Planning",),
        relevant_evidence=(
            "A node is added when the projected storage usage exceeds "
            "seventy percent of the usable capacity.",
        ),
    ),
    Query(
        text="How is an incident declared and who coordinates the response?",
        relevant_sections=("On-Call Rotation",),
        relevant_evidence=(
            "An incident is declared when a service level objective is at "
            "risk.",
        ),
    ),
    Query(
        text="On which day are the invoices issued and how is a partial month "
        "handled?",
        relevant_sections=("Invoice Generation",),
        relevant_evidence=(
            "Invoices are issued on the first day of every month and cover "
            "the calendar month that just ended.",
            "An invoice for a partial month is pro-rated by day, so a "
            "contract that starts on the fifteenth is billed for sixteen "
            "days in that month.",
        ),
    ),
    Query(
        text="How long does a refund take and which invoice number does it "
        "keep?",
        relevant_sections=("Credits And Refunds",),
        relevant_evidence=(
            "A refund is issued within five business days of an approved "
            "credit note.",
            "A partial refund keeps the original invoice number and "
            "reduces the amount that was paid.",
        ),
        notes="the query mentions the invoice number, which also appears in "
        "the invoice generation section",
    ),
    Query(
        text="What happens to a dimension that is not in the rate card?",
        relevant_sections=("Usage Rating",),
        relevant_evidence=(
            "A dimension that is not mentioned in the rate card is billed "
            "at the default rate of the region.",
        ),
    ),
    Query(
        text="Which tax rate applies and how is an exemption recognized?",
        relevant_sections=("Tax Handling",),
        relevant_evidence=(
            "The tax rate that applies is the one in force on the invoice "
            "date.",
            "An exempt customer is identified by a valid exemption "
            "certificate that is stored with the account.",
        ),
    ),
    Query(
        text="How was the coarse-to-fine hierarchy compared with a flat "
        "search?",
        relevant_sections=("Method",),
        relevant_evidence=(
            "Each document is split into a hierarchy of two levels, a "
            "section level and a passage level.",
            "Queries are answered by selecting the relevant sections first "
            "and the passages inside them second.",
            "We compare this with a flat search over all passages.",
        ),
        notes="research-style document; 'hierarchy' also appears in the "
        "abstract and the results",
    ),
    Query(
        text="What did the study find about the number of candidate passages?",
        relevant_sections=("Results",),
        relevant_evidence=(
            "The coarse-to-fine hierarchy reduced the number of candidate "
            "passages that had to be scored by an order of magnitude.",
        ),
    ),
    Query(
        text="Which threats to validity does the study mention?",
        relevant_sections=("Threats To Validity",),
        relevant_evidence=(
            "The corpus is a single technical documentation set, so the "
            "result may not generalize to other domains.",
            "The encoder was not tuned for the hierarchy, which probably "
            "understates the benefit.",
        ),
    ),
)


__all__ = ["CORPUS", "QUERIES", "Document", "Query", "Section"]
