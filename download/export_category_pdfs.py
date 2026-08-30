#!/usr/bin/env python3

"""Download PDF contribution attachments from an Indico category or a single event.

Two modes are supported:

  Category mode (original behaviour):
    --categoryNumber <id> --startDate <YYYY-MM-DD> [--endDate <YYYY-MM-DD>]

  Event mode (single conference / large event, any Indico server):
    --eventId <id>            # uses --indico_site for the server
    --event_url <full-url>    # e.g. https://agenda.infn.it/event/47074/
                              # (indico_site is inferred from the URL)

In event mode no date range is required and credentials are optional for
publicly-accessible events.  For large events the Indico REST API v1 is used
with automatic pagination; the legacy export endpoint is the fallback for older
Indico instances that do not expose the REST API.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


def build_indico_request(
    path,
    params,
    api_token=None,
    api_key=None,
    secret_key=None,
    only_public=False,
    persistent=False,
):
    items = list(params.items()) if hasattr(params, "items") else list(params)
    if only_public:
        items.append(("onlypublic", "yes"))
    # Token mode uses the Authorization header only; do not add legacy auth params.
    if api_key and not api_token:
        items.append(("apikey", api_key))
    if secret_key and not api_token:
        if not persistent:
            import time

            items.append(("timestamp", str(int(time.time()))))
        items = sorted(items, key=lambda item: item[0].lower())
        url = f"{path}?{urllib.parse.urlencode(items)}"
        signature = hmac.new(secret_key.encode("utf-8"), url.encode("utf-8"), hashlib.sha1).hexdigest()
        items.append(("signature", signature))
    if not items:
        return path
    return f"{path}?{urllib.parse.urlencode(items)}"


def resolve_auth_mode(api_token, api_key, secret_key):
    token = (api_token or "").strip()
    key = (api_key or "").strip()
    secret = (secret_key or "").strip()

    if token and (key or secret):
        raise SystemExit("Use either --API_TOKEN or --API_KEY/--SECRET_KEY, not both in the same run.")
    if token:
        return token, "", ""
    if key and secret:
        return "", key, secret
    return token, key, secret


def normalize_end_date(end_date):
    value = (end_date or "").strip()
    if value.lower() == "today":
        return datetime.now().strftime("%Y-%m-%d")
    return value


def load_credentials(api_token, api_key, secret_key, require=True):
    api_token, api_key, secret_key = resolve_auth_mode(api_token, api_key, secret_key)
    if api_token or (api_key and secret_key):
        return api_token, api_key, secret_key

    secret_file = Path("~/.indico-secret-key").expanduser()
    if secret_file.is_file():
        try:
            lines = secret_file.read_text(encoding="utf-8").split()
            if len(lines) >= 2:
                return api_token, lines[0], lines[1]
        except OSError:
            pass

    if require:
        raise SystemExit(
            "Missing Indico credentials. Pass --API_TOKEN or both --API_KEY and --SECRET_KEY, "
            "or populate ~/.indico-secret-key."
        )
    return "", "", ""


def make_request(url, api_token=None, accept=None):
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    if accept:
        headers["Accept"] = accept

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        return response.read()


def fetch_json(path, params, indico_site, api_token=None, api_key=None, secret_key=None):
    url = indico_site + build_indico_request(path, params, api_token, api_key, secret_key)
    payload = make_request(url, api_token=api_token, accept="application/json")
    return json.loads(payload.decode("utf-8"))


def build_authenticated_download_url(url, api_token=None, api_key=None, secret_key=None):
    # Token mode authenticates through Authorization header.
    if api_token:
        return url

    # Key/secret mode requires a signed URL for attachment downloads.
    if api_key and secret_key:
        parsed = urllib.parse.urlsplit(url)
        params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        signed_path_and_query = build_indico_request(
            parsed.path,
            params,
            api_key=api_key,
            secret_key=secret_key,
        )
        signed_path, _, signed_query = signed_path_and_query.partition("?")
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, signed_path, signed_query, parsed.fragment))

    return url


def fetch_category_events(
    category_number, start_date, end_date, indico_site, api_token=None, api_key=None, secret_key=None
):
    reply = fetch_json(
        f"/export/categ/{category_number}.json",
        {
            "from": start_date,
            "to": end_date,
            "pretty": "yes",
            "order": "start",
        },
        indico_site,
        api_token=api_token,
        api_key=api_key,
        secret_key=secret_key,
    )
    return reply.get("results", [])


def fetch_event_details(event_id, indico_site, api_token=None, api_key=None, secret_key=None):
    reply = fetch_json(
        f"/export/event/{int(event_id)}.json",
        {"pretty": "yes", "detail": "subcontributions"},
        indico_site,
        api_token=api_token,
        api_key=api_key,
        secret_key=secret_key,
    )
    results = reply.get("results", [])
    if not results:
        raise ValueError(f"No event details returned for event {event_id}")
    return results[0]


# ---------------------------------------------------------------------------
# Event-mode helpers (direct single-event export, any Indico server)
# ---------------------------------------------------------------------------


def parse_event_url(url):
    """Return ``(base_url, event_id)`` from a URL like ``https://agenda.infn.it/event/47074/``."""
    parsed = urllib.parse.urlsplit(url)
    m = re.search(r"/event/(\d+)", parsed.path)
    if not m:
        raise ValueError(f"Cannot extract event ID from URL: {url}")
    return f"{parsed.scheme}://{parsed.netloc}", int(m.group(1))


def _normalize_download_url(url, indico_site):
    """Make a relative attachment download URL absolute."""
    if url and not url.startswith("http"):
        return indico_site.rstrip("/") + (url if url.startswith("/") else "/" + url)
    return url


def fetch_contributions_rest(event_id, indico_site, api_token=None, per_page=100, debug=False):
    """Fetch all contributions via the Indico REST API v1 with automatic pagination.

    Handles large events (hundreds of contributions) by requesting pages of
    *per_page* contributions at a time.  Relative ``download_url`` fields in
    attachment objects are made absolute before returning.
    """
    all_contribs: list = []
    page = 1
    base = indico_site.rstrip("/")
    while True:
        url = f"{base}/api/v1/events/{event_id}/contributions/?page={page}&per_page={per_page}"
        if debug:
            print(f"Fetching contributions (REST API page {page}): {url}")
        raw = make_request(url, api_token=api_token, accept="application/json")
        data = json.loads(raw.decode("utf-8"))
        batch = data.get("data", [])
        for contrib in batch:
            for folder in contrib.get("folders", []):
                for att in folder.get("attachments", []):
                    if "download_url" in att:
                        att["download_url"] = _normalize_download_url(att["download_url"], indico_site)
        all_contribs.extend(batch)
        if page >= data.get("pages", 1) or not batch:
            break
        page += 1
    return all_contribs


def fetch_event_for_export(event_id, indico_site, api_token=None, api_key=None, secret_key=None, debug=False):
    """Return an event-details dict suitable for :func:`collect_contribution_pdfs`.

    Strategy
    --------
    1. Try the **Indico REST API v1** (paginated, recommended for large events).
       Basic event metadata (title, date, URL) is fetched separately via the
       lightweight export endpoint so the heavy ``detail=subcontributions``
       payload is avoided.
    2. If the REST API is unavailable (HTTP 404/405 — Indico < 3.0) or returns
       any non-auth error, fall back to the **legacy export API** which returns
       everything in one request.

    Authentication errors (HTTP 401/403) are always propagated so the caller
    receives a meaningful message rather than a misleading fallback failure.
    """
    # --- Attempt 1: paginated REST API ---
    try:
        contribs = fetch_contributions_rest(event_id, indico_site, api_token=api_token, debug=debug)
        if debug:
            print(f"REST API: fetched {len(contribs)} contributions")

        # Lightweight event-info fetch (no contribution detail).
        event_info: dict = {}
        try:
            reply = fetch_json(
                f"/export/event/{int(event_id)}.json",
                {"pretty": "yes"},
                indico_site,
                api_token=api_token,
                api_key=api_key,
                secret_key=secret_key,
            )
            results = reply.get("results", [])
            if results:
                event_info = results[0]
        except Exception:
            pass

        event_info["contributions"] = contribs
        return event_info

    except urllib.error.HTTPError as rest_exc:
        if rest_exc.code in (401, 403):
            raise  # Authentication error — do not swallow.
        if debug:
            print(f"REST API not available (HTTP {rest_exc.code}); falling back to legacy export API.")
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as rest_exc:
        if debug:
            print(f"REST API failed ({rest_exc}); falling back to legacy export API.")

    # --- Attempt 2: legacy export API ---
    reply = fetch_json(
        f"/export/event/{int(event_id)}.json",
        {"pretty": "yes", "detail": "subcontributions"},
        indico_site,
        api_token=api_token,
        api_key=api_key,
        secret_key=secret_key,
    )
    results = reply.get("results", [])
    if not results:
        raise ValueError(f"No event details returned for event {event_id}")
    return results[0]


def is_pdf_attachment(attachment):
    content_type = (
        attachment.get("content_type") or attachment.get("mimeType") or attachment.get("mime_type") or ""
    ).lower()
    title = (attachment.get("title") or "").lower()
    file_name = (attachment.get("fileName") or attachment.get("filename") or "").lower()
    download_url = (attachment.get("download_url") or "").lower()
    return (
        content_type == "application/pdf"
        or title.endswith(".pdf")
        or file_name.endswith(".pdf")
        or ".pdf" in download_url
    )


def normalize_speakers(speakers):
    normalized = []
    for speaker in speakers:
        full_name = speaker.get("fullName")
        if not full_name:
            full_name = " ".join(part for part in [speaker.get("first_name"), speaker.get("last_name")] if part).strip()
        normalized.append(
            {
                "full_name": full_name,
                "affiliation": speaker.get("affiliation"),
            }
        )
    return normalized


def make_unique_pdf_name(event_date, event_id, contribution_id, attachment_title, download_url):
    safe_date = event_date or "unknown-date"
    stem = "{}-{}-{}-{}".format(
        safe_date,
        event_id,
        contribution_id,
        hashlib.sha256((download_url + attachment_title).encode("utf-8")).hexdigest()[:12],
    )
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem) + ".pdf"


def write_pdf_metadata(records, metadata_dir):
    metadata_dir.mkdir(parents=True, exist_ok=True)
    written_files = []
    for record in records:
        pdf_stem = Path(record["pdf_file"]).stem
        file_name = re.sub(r"[^A-Za-z0-9._-]", "_", pdf_stem) + ".json"
        metadata_path = metadata_dir / file_name
        payload = dict(record)
        payload["metadata_file"] = file_name
        payload["metadata_path"] = str(metadata_path)
        metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written_files.append(metadata_path)

    return sorted(written_files)


def collect_contribution_pdfs(event_details, category_number):
    meeting_date = (event_details.get("startDate") or {}).get("date")
    meeting_title = event_details.get("title")
    meeting_url = event_details.get("url")
    event_id = event_details.get("id")

    for contribution in event_details.get("contributions", []):
        contribution_id = contribution.get("id") or contribution.get("db_id")
        contribution_title = contribution.get("title")
        speakers = normalize_speakers(contribution.get("speakers", []))

        for folder in contribution.get("folders", []):
            folder_title = folder.get("title")
            for attachment in folder.get("attachments", []):
                if not is_pdf_attachment(attachment):
                    continue

                download_url = attachment.get("download_url")
                if not download_url:
                    continue

                file_name = make_unique_pdf_name(
                    meeting_date,
                    event_id,
                    contribution_id,
                    attachment.get("title") or "attachment",
                    download_url,
                )

                yield {
                    "category_id": category_number,
                    "meeting_id": event_id,
                    "meeting_title": meeting_title,
                    "meeting_url": meeting_url,
                    "date": meeting_date,
                    "contribution_id": contribution_id,
                    "contribution_title": contribution_title,
                    "contribution_url": contribution.get("url"),
                    "authors": speakers,
                    "author": ", ".join(speaker["full_name"] for speaker in speakers if speaker.get("full_name")),
                    "attachment_title": attachment.get("title"),
                    "attachment_id": attachment.get("id"),
                    "attachment_is_protected": bool(attachment.get("is_protected")),
                    "attachment_folder": folder_title,
                    "attachment_size": attachment.get("size"),
                    "download_url": download_url,
                    "pdf_file": file_name,
                }


def download_pdf(url, destination, api_token=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = make_request(url, api_token=api_token)
    if not payload.lstrip().startswith(b"%PDF-"):
        snippet = payload[:120].decode("utf-8", errors="replace")
        raise ValueError(f"Downloaded content is not a PDF. Response starts with: {snippet!r}")
    destination.write_bytes(payload)


def existing_file_is_pdf(path):
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
        return header.lstrip().startswith(b"%PDF-")
    except OSError:
        return False


def export_category_pdfs(
    *,
    category_number,
    start_date,
    output_dir,
    metadata_dir,
    end_date,
    api_token="",
    api_key="",
    secret_key="",
    indico_site="https://indico.cern.ch",
    overwrite=False,
    debug=False,
):
    api_token, api_key, secret_key = load_credentials(api_token, api_key, secret_key)
    end_date = normalize_end_date(end_date)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_dir)
    metadata_path.mkdir(parents=True, exist_ok=True)

    # Pre-scan the output directory once to avoid N individual stat() RPCs on the network
    # filesystem. Build {filename: is_valid_pdf} so the per-record loop does zero file I/O
    # for existence and validity checks.
    existing_pdf_validity: dict = (
        {} if overwrite else {p.name: existing_file_is_pdf(p) for p in output_path.iterdir() if p.is_file()}
    )

    events = fetch_category_events(
        category_number,
        start_date,
        end_date,
        indico_site,
        api_token=api_token,
        api_key=api_key,
        secret_key=secret_key,
    )

    records = []
    for event in events:
        event_id = event.get("id")
        if debug:
            print(f"Fetching event {event_id}")
        event_details = fetch_event_details(
            event_id,
            indico_site,
            api_token=api_token,
            api_key=api_key,
            secret_key=secret_key,
        )

        for record in collect_contribution_pdfs(event_details, category_number):
            destination = output_path / record["pdf_file"]
            file_exists = record["pdf_file"] in existing_pdf_validity
            should_download = overwrite or not file_exists
            if file_exists and not overwrite and not existing_pdf_validity[record["pdf_file"]]:
                # Recover from prior runs that saved HTML auth pages with .pdf extension.
                should_download = True

            if should_download:
                if record.get("attachment_is_protected") and not api_token:
                    raise ValueError(
                        "Attachment '{}' is protected and cannot be downloaded with API_KEY/SECRET_KEY. "
                        "Please rerun with --API_TOKEN from /user/tokens/.".format(record.get("attachment_title"))
                    )
                authenticated_url = build_authenticated_download_url(
                    record["download_url"],
                    api_token=api_token,
                    api_key=api_key,
                    secret_key=secret_key,
                )
                if debug:
                    print(f"Downloading {authenticated_url} -> {destination}")
                download_pdf(authenticated_url, destination, api_token=api_token)
            record["pdf_path"] = str(destination)
            records.append(record)

    records.sort(
        key=lambda item: (
            item.get("date") or "",
            item.get("meeting_title") or "",
            item.get("contribution_title") or "",
            item.get("attachment_title") or "",
        )
    )

    metadata_files = write_pdf_metadata(records, metadata_path)
    return records, metadata_files


def export_event_pdfs(
    *,
    event_id,
    output_dir,
    metadata_dir,
    api_token="",
    api_key="",
    secret_key="",
    indico_site="https://indico.cern.ch",
    overwrite=False,
    debug=False,
):
    """Download PDF attachments from a single Indico event on any Indico server.

    Unlike :func:`export_category_pdfs`, no date range is required.  Credentials
    are optional for publicly-accessible events.  Large events with many
    contributions are handled transparently via the Indico REST API v1 (paginated);
    the legacy export API is used as a fallback for older Indico instances.
    """
    api_token, api_key, secret_key = load_credentials(api_token, api_key, secret_key, require=False)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_dir)
    metadata_path.mkdir(parents=True, exist_ok=True)

    existing_pdf_validity: dict = (
        {} if overwrite else {p.name: existing_file_is_pdf(p) for p in output_path.iterdir() if p.is_file()}
    )

    if debug:
        print(f"Fetching event {event_id} from {indico_site}")

    event_details = fetch_event_for_export(
        event_id,
        indico_site,
        api_token=api_token,
        api_key=api_key,
        secret_key=secret_key,
        debug=debug,
    )

    records = []
    for record in collect_contribution_pdfs(event_details, category_number=None):
        destination = output_path / record["pdf_file"]
        file_exists = record["pdf_file"] in existing_pdf_validity
        should_download = overwrite or not file_exists
        if file_exists and not overwrite and not existing_pdf_validity[record["pdf_file"]]:
            should_download = True

        if should_download:
            if record.get("attachment_is_protected") and not api_token:
                raise ValueError(
                    "Attachment '{}' is protected and cannot be downloaded without "
                    "authentication.  Please provide --API_TOKEN.".format(record.get("attachment_title"))
                )
            authenticated_url = build_authenticated_download_url(
                record["download_url"],
                api_token=api_token,
                api_key=api_key,
                secret_key=secret_key,
            )
            if debug:
                print(f"Downloading {authenticated_url} -> {destination}")
            download_pdf(authenticated_url, destination, api_token=api_token)
        record["pdf_path"] = str(destination)
        records.append(record)

    records.sort(
        key=lambda item: (
            item.get("date") or "",
            item.get("contribution_title") or "",
            item.get("attachment_title") or "",
        )
    )

    metadata_files = write_pdf_metadata(records, metadata_path)
    return records, metadata_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--categoryNumber",
        type=int,
        help="Indico category id to scan (category mode; requires --startDate)",
    )
    source_group.add_argument(
        "--eventId",
        type=int,
        help="Indico event id to export directly (event mode; uses --indico_site)",
    )
    source_group.add_argument(
        "--event_url",
        help=(
            "Full URL of an Indico event, e.g. https://agenda.infn.it/event/47074/ "
            "(event mode; --indico_site is inferred from the URL)"
        ),
    )

    parser.add_argument(
        "--startDate",
        default="",
        help="Earliest meeting date to include (category mode only), e.g. 2024-01-01",
    )
    parser.add_argument(
        "--endDate",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Latest meeting date to include (category mode only). Use YYYY-MM-DD (default: current date).",
    )
    parser.add_argument(
        "--outputDir",
        required=True,
        help="Directory where PDF files will be written",
    )
    parser.add_argument(
        "--metadataDir",
        default="",
        help="Directory where one metadata JSON file per PDF is written. Defaults to <outputDir>/metadata/",
    )
    parser.add_argument(
        "--API_TOKEN",
        default=os.getenv("INDICO_API_TOKEN", ""),
        help="Indico personal access token from /user/tokens/",
    )
    parser.add_argument(
        "--API_KEY",
        default=os.getenv("INDICO_API_KEY", ""),
        help="Legacy Indico API key from /user/api/",
    )
    parser.add_argument(
        "--SECRET_KEY",
        default=os.getenv("INDICO_SECRET_KEY", ""),
        help="Legacy Indico secret key from /user/api/",
    )
    parser.add_argument(
        "--indico_site",
        default="https://indico.cern.ch",
        help="Base URL of the Indico instance (ignored when --event_url is used)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload PDFs even if the output file already exists",
    )
    parser.add_argument("--debug", action="store_true", help="Print progress information")
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()

    # Resolve the Indico server and event/category ID from the arguments.
    indico_site = args.indico_site
    event_id = None
    if args.event_url:
        indico_site, event_id = parse_event_url(args.event_url)
    elif args.eventId is not None:
        event_id = args.eventId

    metadata_dir = args.metadataDir or os.path.join(args.outputDir, "metadata")

    try:
        if event_id is not None:
            # Event mode: export a single event directly.
            records, metadata_files = export_event_pdfs(
                event_id=event_id,
                output_dir=args.outputDir,
                metadata_dir=metadata_dir,
                api_token=args.API_TOKEN,
                api_key=args.API_KEY,
                secret_key=args.SECRET_KEY,
                indico_site=indico_site,
                overwrite=args.overwrite,
                debug=args.debug,
            )
        else:
            # Category mode: original behaviour.
            if not args.startDate:
                parser.error("--startDate is required when using --categoryNumber.")
            records, metadata_files = export_category_pdfs(
                category_number=args.categoryNumber,
                start_date=args.startDate,
                end_date=args.endDate,
                output_dir=args.outputDir,
                metadata_dir=metadata_dir,
                api_token=args.API_TOKEN,
                api_key=args.API_KEY,
                secret_key=args.SECRET_KEY,
                indico_site=indico_site,
                overwrite=args.overwrite,
                debug=args.debug,
            )
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            body = ""
        detail = f"Indico request failed: {error}"
        if body:
            detail += f"\nResponse body: {body}"
        raise SystemExit(detail) from error
    except ValueError as error:
        raise SystemExit(str(error)) from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Network error while contacting Indico: {error}") from error

    print(f"Exported {len(records)} PDF attachment(s)")
    print(f"Wrote {len(metadata_files)} PDF metadata file(s) in {metadata_dir}")


if __name__ == "__main__":
    main()
