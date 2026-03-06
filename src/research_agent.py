from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup


@dataclass
class ResearchResult:
    source: str
    title: str
    url: str
    snippet: str


@dataclass
class CompanyBackground:
    cin: str
    incorporation_date: str
    status: str
    paid_up_capital: float
    directors: List[str]


@dataclass
class LegalRecord:
    court_name: str
    case_type: str
    plaintiff: str
    status: str
    filing_date: str


def fetch_mca_background(company_name: str) -> CompanyBackground | None:
    """
    Simulates fetching company background from the Ministry of Corporate Affairs (MCA).
    """
    if not company_name:
        return None
        
    length = len(company_name)
    return CompanyBackground(
        cin=f"U{10000 + length}TG201{length % 10}PTC{200000 + length}",
        incorporation_date=f"201{length % 10}-05-12",
        status="Active" if length % 2 == 0 else "Active (Non-Compliant)",
        paid_up_capital=100000.0 * length,
        directors=["Director A", "Director B"] if length > 5 else ["Director A"]
    )


def fetch_ecourts_litigation(company_name: str) -> List[LegalRecord]:
    """
    Simulates hitting the e-Courts web services to check for pending litigation.
    """
    records = []
    if "industries" in company_name.lower() or "faketech" in company_name.lower():
        records.append(LegalRecord(
            court_name="NCLT Mumbai",
            case_type="Insolvency Petition",
            plaintiff="Operational Creditor Pvt Ltd",
            status="Pending Admission",
            filing_date="2024-01-15"
        ))
    if "trading" in company_name.lower():
        records.append(LegalRecord(
            court_name="High Court of Delhi",
            case_type="Commercial Suit Recovery",
            plaintiff="Supplier Bank Ltd",
            status="Pending",
            filing_date="2023-11-20"
        ))
    return records


def fetch_news_headlines(query: str, max_results: int = 10) -> List[ResearchResult]:
    """
    Very simple news search using a generic web search endpoint.
    NOTE: This is a prototype and may need to be adapted to specific APIs
    (e.g. NewsAPI, custom search) in production.
    """
    results: List[ResearchResult] = []
    try:
        resp = requests.get(
            "https://news.google.com/search",
            params={"q": query, "hl": "en-IN"},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception:
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    articles = soup.select("article")[:max_results]
    for art in articles:
        title_el = art.select_one("h3")
        if not title_el or not title_el.text:
            continue
        link_el = title_el.find("a")
        url = ""
        if link_el and link_el.get("href"):
            href = link_el["href"]
            if href.startswith("./"):
                url = "https://news.google.com" + href[1:]
            else:
                url = href
        snippet = ""
        results.append(
            ResearchResult(
                source="news",
                title=title_el.text.strip(),
                url=url,
                snippet=snippet,
            )
        )

    return results


def summarize_research(company_name: str, sector: str | None = None) -> Dict[str, Any]:
    """
    High-level wrapper to collect external research signals.
    Currently only news headlines; MCA/e-Courts can be added similarly.
    """
    query_parts = [company_name]
    if sector:
        query_parts.append(sector)
    query = " ".join(query_parts)
    headlines = fetch_news_headlines(query)

    titles = [h.title for h in headlines]
    urls = [h.url for h in headlines]

    # Very lightweight risk heuristics over news titles
    litigation_words = ["litigation", "lawsuit", "suit", "case", "dispute", "tribunal"]
    headwind_words = ["slowdown", "crisis", "headwind", "pressure", "stress", "default", "downgrade", "insolvency"]

    litigation_news_count = 0
    headwind_hits = 0
    for t in titles:
        tl = t.lower()
        if any(w in tl for w in litigation_words):
            litigation_news_count += 1
        if any(w in tl for w in headwind_words):
            headwind_hits += 1

    headline_count = len(titles)
    sector_headwind_score = float(headwind_hits / headline_count) if headline_count else 0.0

    # 2. MCA Fetch
    mca_data = fetch_mca_background(company_name)
    mca_status = mca_data.status if mca_data else "Unknown"
    mca_directors = len(mca_data.directors) if mca_data else 0
    
    # MCA heuristics 
    mca_risk_flag = 1.0 if "Non-Compliant" in mca_status or "Strike Off" in mca_status else 0.0

    # 3. e-Courts Fetch
    litigation_records = fetch_ecourts_litigation(company_name)
    ecourts_litigation_count = len(litigation_records)
    
    # Look for severe insolvency cases
    severe_cases = sum(1 for r in litigation_records if "Insolvency" in r.case_type or "NCLT" in r.court_name)
    ecourts_severe_risk = 1.0 if severe_cases > 0 else 0.0

    return {
        "research_news_headline_count": headline_count,
        "research_news_titles": titles[:5],
        "research_news_urls": urls[:5],
        "research_litigation_news_count": litigation_news_count,
        "research_sector_headwind_score": sector_headwind_score,
        "research_mca_status": mca_status,
        "research_mca_directors_count": mca_directors,
        "research_mca_risk_flag": mca_risk_flag,
        "research_ecourts_litigation_count": ecourts_litigation_count,
        "research_ecourts_severe_risk": ecourts_severe_risk,
    }

