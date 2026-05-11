from pathlib import Path
import argparse
from lxml import etree

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


def strip_namespace(tag: str) -> str:
    """Remove XML namespace from tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def element_to_dict(element):
    """Convert direct child elements to a flat dictionary."""
    data = {}
    for child in element:
        tag = strip_namespace(child.tag)
        text = clean_text(child.text)
        if text:
            data[tag] = text
    return data


def find_elements_by_names(root, possible_names):
    """
    Find elements where the local tag name matches one of possible_names.
    Works even when XML namespaces are present.
    """
    matches = []
    possible_names = {name.lower() for name in possible_names}

    for element in root.iter():
        local_name = strip_namespace(element.tag).lower()
        if local_name in possible_names:
            matches.append(element)

    return matches


def make_table(title, records, preferred_columns=None, max_rows=200):
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph(title, styles["Heading2"]))

    if not records:
        story.append(Paragraph("No data found in this section.", styles["Normal"]))
        story.append(Spacer(1, 0.4 * cm))
        return story

    records = records[:max_rows]

    all_columns = []
    for record in records:
        for key in record.keys():
            if key not in all_columns:
                all_columns.append(key)

    if preferred_columns:
        columns = [c for c in preferred_columns if c in all_columns]
        columns += [c for c in all_columns if c not in columns]
    else:
        columns = all_columns

    # Keep tables readable
    columns = columns[:8]

    table_data = [columns]

    for record in records:
        row = []
        for column in columns:
            value = clean_text(record.get(column, ""))
            if len(value) > 80:
                value = value[:77] + "..."
            row.append(value)
        table_data.append(row)

    table = Table(table_data, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ]
        )
    )

    story.append(table)

    if len(records) == max_rows:
        story.append(Spacer(1, 0.2 * cm))
        story.append(
            Paragraph(
                f"Only the first {max_rows} rows are shown in this section.",
                styles["Normal"],
            )
        )

    story.append(Spacer(1, 0.7 * cm))
    return story


def parse_xaf(xaf_path):
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xaf_path), parser)
    root = tree.getroot()

    # These names cover common Dutch XML Auditfile structures.
    company_elements = find_elements_by_names(
        root, ["company", "companyinfo", "header", "administration"]
    )

    ledger_elements = find_elements_by_names(
        root, ["ledgeraccount", "account", "generalledgeraccount"]
    )

    transaction_elements = find_elements_by_names(
        root, ["transaction", "entry", "journalentry", "line", "transactionline"]
    )

    customer_elements = find_elements_by_names(
        root, ["customer", "debtor", "debtorinfo"]
    )

    supplier_elements = find_elements_by_names(
        root, ["supplier", "creditor", "creditorinfo"]
    )

    vat_elements = find_elements_by_names(root, ["vatcode", "taxcode", "vat", "tax"])

    return {
        "company": [element_to_dict(e) for e in company_elements],
        "ledger": [element_to_dict(e) for e in ledger_elements],
        "transactions": [element_to_dict(e) for e in transaction_elements],
        "customers": [element_to_dict(e) for e in customer_elements],
        "suppliers": [element_to_dict(e) for e in supplier_elements],
        "vat": [element_to_dict(e) for e in vat_elements],
    }


def build_pdf(xaf_path, output_pdf):
    data = parse_xaf(xaf_path)

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(A4),
        rightMargin=1 * cm,
        leftMargin=1 * cm,
        topMargin=1 * cm,
        bottomMargin=1 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=12
    )

    story = []

    story.append(Paragraph("XAF Audit File Report", title_style))
    story.append(Paragraph(f"Source file: {xaf_path.name}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Summary", styles["Heading2"]))

    summary_data = [
        ["Section", "Records found"],
        ["Company/header records", str(len(data["company"]))],
        ["Ledger accounts", str(len(data["ledger"]))],
        ["Transactions / journal lines", str(len(data["transactions"]))],
        ["Customers / debtors", str(len(data["customers"]))],
        ["Suppliers / creditors", str(len(data["suppliers"]))],
        ["VAT / tax records", str(len(data["vat"]))],
    ]

    summary_table = Table(summary_data, colWidths=[8 * cm, 5 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )

    story.append(summary_table)
    story.append(PageBreak())

    story += make_table(
        "Company / Header Information",
        data["company"],
        preferred_columns=["companyName", "name", "fiscalYear", "startDate", "endDate"],
    )

    story += make_table(
        "Ledger Accounts",
        data["ledger"],
        preferred_columns=["accountID", "accountCode", "accountDesc", "description", "balance"],
    )

    story += make_table(
        "Transactions / Journal Lines",
        data["transactions"],
        preferred_columns=[
            "transactionID",
            "journalID",
            "period",
            "date",
            "accountID",
            "accountCode",
            "description",
            "debit",
            "credit",
            "amount",
        ],
        max_rows=500,
    )

    story += make_table(
        "Customers / Debtors",
        data["customers"],
        preferred_columns=["customerID", "name", "city", "country", "vatNumber"],
    )

    story += make_table(
        "Suppliers / Creditors",
        data["suppliers"],
        preferred_columns=["supplierID", "name", "city", "country", "vatNumber"],
    )

    story += make_table(
        "VAT / Tax Codes",
        data["vat"],
        preferred_columns=["vatCode", "taxCode", "description", "percentage", "rate"],
    )

    doc.build(story)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Convert an XAF XML audit file into a searchable PDF report."
    )

    arg_parser.add_argument("input_xaf", help="Path to the .xaf or .xml audit file")

    arg_parser.add_argument(
        "-o",
        "--output",
        help="Output PDF path. Default: same name as input file with .pdf extension",
    )

    args = arg_parser.parse_args()

    xaf_path = Path(args.input_xaf)

    if not xaf_path.exists():
        raise FileNotFoundError(f"Input file not found: {xaf_path}")

    output_pdf = Path(args.output) if args.output else xaf_path.with_suffix(".pdf")

    build_pdf(xaf_path, output_pdf)

    print(f"PDF created: {output_pdf}")


if __name__ == "__main__":
    main()

