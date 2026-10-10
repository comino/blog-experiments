#!/usr/bin/env python3
"""Prepare a versioned, deterministic ClickHouse SQL benchmark (no model calls)."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:18123/"
DATABASE = "exp04_budget"


def query(sql, database="default"):
    response = requests.post(URL, params={"database": database}, data=sql.encode(), timeout=90)
    response.raise_for_status()
    return response.text


def build_questions(source):
    questions = json.loads((source / "questions.json").read_text())
    questions += json.loads((source / "questions_t6.json").read_text())
    columns = [
        "event count", "all events columns in schema order", "event count", "event_type", "event count",
        "all events columns in schema order", "user count", "earliest timestamp, latest timestamp",
        "all events columns in schema order", "event count",
        "event_type, event count", "device, average duration", "day as Date, event count",
        "country, event count", "hour number, event count", "average duration", "month as Date, event count",
        "event_type, distinct country count", "country, total revenue", "purchase count, pageview count",
        "median duration", "exact distinct user count", "all events columns in schema order",
        "event_id, source", "tag, occurrence count", "all events columns in schema order",
        "event_type, p90, p99", "mobile revenue", "property key", "all events columns in schema order",
        "event_id, timestamp, event_type, plan", "plan, event count", "user_id, email",
        "user_id, event count, plan", "user_id, event count", "all events columns in schema order",
        "plan, average events per active user", "user_id, email, created_at", "country, pro user count",
        "device, user_id, event count", "user_id, event_id, timestamp, row number",
        "day as Date, daily event count, running total", "day as Date, event count, previous-day count",
        "event_id, uppercase tags array", "event_type, user_id, total duration",
        "day as Date, event count", "user_id, sorted distinct event types array",
        "user_id, last event_id, last timestamp", "hour as DateTime, event count",
        "user_id, sorted distinct premium tags array", "users_l1, users_l2, users_l3", "r1, r2, r3",
        "user count", "user_id, first_event_time", "event_type, event_id, timestamp", "day as Date, revenue",
        "array of four counts", "cumulative counts array", "country, page", "exact distinct property key count",
    ]
    orders = {
        "t1_02": "timestamp descending, then event_id descending; exactly 10 rows",
        "t1_04": "event_type ascending", "t1_06": "timestamp ascending, then event_id ascending; at most 5 rows",
        "t1_09": "timestamp ascending, then event_id ascending",
        "t2_01": "event count descending, then event_type ascending",
        "t2_02": "average duration descending, then device ascending", "t2_03": "day ascending",
        "t2_04": "event count descending, then country ascending", "t2_05": "hour ascending",
        "t2_07": "month ascending", "t2_08": "distinct country count descending, then event_type ascending",
        "t2_09": "total revenue descending, then country ascending",
        "t3_03": "timestamp ascending, then event_id ascending; at most 100 rows",
        "t3_04": "timestamp ascending, then event_id ascending; at most 10 rows",
        "t3_05": "occurrence count descending, then tag ascending",
        "t3_06": "timestamp ascending, then event_id ascending; at most 100 rows",
        "t3_07": "event_type ascending", "t3_09": "property key ascending",
        "t3_10": "timestamp ascending, then event_id ascending; at most 100 rows",
        "t4_01": "timestamp ascending, then event_id ascending; at most 100 rows",
        "t4_02": "event count descending, then plan ascending", "t4_03": "user_id ascending",
        "t4_04": "event count descending, then user_id ascending; at most 10 users",
        "t4_05": "user_id ascending", "t4_06": "timestamp ascending, then event_id ascending; at most 100 rows",
        "t4_07": "average events per active user descending, then plan ascending", "t4_08": "user_id ascending",
        "t4_09": "pro user count descending, then country ascending; exactly 1 row",
        "t4_10": "device ascending, then user_id ascending; include all users tied for the maximum",
        "t5_01": "user_id ascending, then row number ascending; at most 100 rows",
        "t5_02": "day ascending", "t5_03": "day ascending", "t5_04": "event_id ascending; exactly 20 rows",
        "t5_05": "event_type ascending", "t5_06": "day ascending",
        "t5_07": "user_id ascending; at most 50 users", "t5_08": "user_id ascending; at most 50 users",
        "t5_09": "hour ascending", "t5_10": "user_id ascending; at most 50 users",
        "t6_04": "user_id ascending", "t6_05": "event_type ascending, timestamp descending, event_id descending",
        "t6_06": "day ascending", "t6_09": "country ascending",
    }
    substitutions = {
        "t1_02": [("timestamp DESC", "timestamp DESC, event_id DESC")],
        "t1_06": [("ORDER BY timestamp", "ORDER BY timestamp, event_id")],
        "t1_09": [("ORDER BY timestamp", "ORDER BY timestamp, event_id")],
        "t2_01": [("cnt DESC", "cnt DESC, event_type")],
        "t2_02": [("avg_duration DESC", "avg_duration DESC, device")],
        "t2_04": [("cnt DESC", "cnt DESC, country")],
        "t2_08": [("uniq(country)", "uniqExact(country)"), ("country_count DESC", "country_count DESC, event_type")],
        "t2_09": [("total_revenue DESC", "total_revenue DESC, country")],
        "t3_01": [("quantile(0.5)", "quantileExact(0.5)")],
        "t3_03": [("ORDER BY timestamp", "ORDER BY timestamp, event_id")],
        "t3_04": [("ORDER BY timestamp", "ORDER BY timestamp, event_id")],
        "t3_05": [("cnt DESC", "cnt DESC, tag")],
        "t3_06": [("ORDER BY timestamp", "ORDER BY timestamp, event_id")],
        "t3_07": [("quantile(", "quantileExact(")],
        "t3_10": [("ORDER BY timestamp", "ORDER BY timestamp, event_id")],
        "t4_01": [("ORDER BY e.timestamp", "ORDER BY e.timestamp, e.event_id")],
        "t4_02": [("cnt DESC", "cnt DESC, u.plan")],
        "t4_04": [("ORDER BY cnt DESC LIMIT", "ORDER BY cnt DESC, user_id LIMIT"), ("ORDER BY t.cnt DESC", "ORDER BY t.cnt DESC, t.user_id")],
        "t4_06": [("ORDER BY timestamp", "ORDER BY timestamp, event_id")],
        "t4_07": [("avg_events_per_user DESC", "avg_events_per_user DESC, plan")],
        "t4_09": [("cnt DESC", "cnt DESC, country")],
        "t4_10": [("ORDER BY device", "ORDER BY device, user_id")],
        "t5_01": [("PARTITION BY user_id ORDER BY timestamp", "PARTITION BY user_id ORDER BY timestamp, event_id")],
        "t5_04": [("FROM events LIMIT", "FROM events ORDER BY event_id LIMIT")],
        "t5_05": [("argMax(user_id, total_dur)", "argMax(user_id, tuple(total_dur, -toInt64(user_id)))")],
        "t5_08": [("argMax(event_id, timestamp)", "argMax(event_id, tuple(timestamp, event_id))")],
        "t6_05": [("timestamp DESC LIMIT", "timestamp DESC, event_id DESC LIMIT")],
        "t6_09": [("argMaxIf(page, timestamp,", "argMaxIf(page, tuple(timestamp, event_id),")],
    }
    for q, cols in zip(questions, columns, strict=True):
        q["original_natural_language"] = q["natural_language"]
        q["original_reference_sql"] = q["reference_sql"]
        for old, new in substitutions.get(q["id"], []):
            assert old in q["reference_sql"], (q["id"], old)
            q["reference_sql"] = q["reference_sql"].replace(old, new)
        q["contract"] = {"columns": cols, "ordered": q["id"] in orders, "order": orders.get(q["id"]), "column_aliases_scored": False, "float_rtol": 1e-6, "float_atol": 1e-6}
        suffix = f"\nOutput contract: return only these columns, in this order: {cols}."
        if q["id"] in orders:
            suffix += f" Order/limit: {orders[q['id']]}."
        q["natural_language"] += suffix
        if q["id"] in ("t3_01", "t3_07"):
            q["natural_language"] += " Use exact quantiles with ClickHouse quantileExact semantics, not approximate sampling."
        if q["id"] == "t4_03":
            q["reference_sql"] = "SELECT user_id, email FROM users WHERE user_id NOT IN (SELECT user_id FROM events) ORDER BY user_id"
        if q["id"] in ("t4_05", "t4_08"):
            q["reference_sql"] += " ORDER BY user_id"
        if q["id"] in ("t4_05", "t4_07"):
            q["natural_language"] += " The average considers only users with at least one event."
        if q["id"] == "t5_01":
            q["natural_language"] += " Within each user, break timestamp ties by event_id ascending."
        if q["id"] == "t5_03":
            q["natural_language"] += " Use zero as the previous count for the first day."
        if q["id"] == "t5_05":
            q["natural_language"] += " Exclude null durations. Break total-duration ties by the smallest user_id."
        if q["id"] == "t5_07":
            q["natural_language"] += " Deduplicate event types within each user's array."
        if q["id"] in ("t5_08", "t6_09"):
            q["natural_language"] += " Break timestamp ties by choosing the greatest event_id."
        if q["id"] == "t5_10":
            q["natural_language"] += " Match the case-sensitive substring 'premium' anywhere in a tag; flatten, deduplicate, and sort each user's matching tags."
            q["reference_sql"] = "SELECT user_id, arraySort(arrayDistinct(arrayFlatten(groupArray(arrayFilter(x -> position(x, 'premium') > 0, tags))))) AS premium_tags FROM events GROUP BY user_id HAVING notEmpty(premium_tags) ORDER BY user_id LIMIT 50"
        q["protocol_version"] = "budget-sql-v1"
    assert len(questions) == 60
    return questions


def create_data():
    query(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    existing = int(query("SELECT count() FROM system.tables WHERE database = 'exp04_budget' AND name IN ('users','events')").strip())
    if existing:
        raise RuntimeError("Benchmark tables already exist; refusing to replace them")
    ddl = (BASE / "schema.sql").read_text()
    for statement in ddl.split(";"):
        if statement.strip():
            query(statement, DATABASE)
    # Independent hashes per field, stable across invocations; no rand() calls.
    users_sql = """INSERT INTO users SELECT
        number + 1,
        toDateTime64('2023-01-01 00:00:00', 3, 'UTC') + toIntervalSecond(cityHash64(number, toUInt64(1)) % (730 * 86400)),
        ['free','pro','enterprise'][1 + cityHash64(number, toUInt64(2)) % 3],
        concat('user',toString(number+1),'@example.com'),
        if(cityHash64(number,toUInt64(3)) % 5 = 0, NULL, toUInt8(18+cityHash64(number,toUInt64(4))%50)),
        ['US','DE','GB','FR','JP','BR','IN','CA','AU','NL'][1+cityHash64(number,toUInt64(5))%10],
        map('signup_source',['organic','paid','referral','social'][1+cityHash64(number,toUInt64(6))%4])
        FROM numbers(10000)"""
    events_sql = """INSERT INTO events SELECT
        number+1,
        if(number < 12,
           toDateTime64('2024-01-01 00:00:00',3,'UTC') + toIntervalSecond(intDiv(number,4)*86400 + (number%4)*10),
           toDateTime64('2024-01-01 00:00:00',3,'UTC') + toIntervalMillisecond(cityHash64(number,toUInt64(20)) % (366*86400000))),
        if(number < 12, toUInt64(42), 1+cityHash64(number,toUInt64(21))%9900),
        if(number < 12, ['signup','pageview','click','purchase'][1+number%4], ['pageview','click','purchase','signup','logout'][1+cityHash64(number,toUInt64(22))%5]),
        concat('/', ['home','about','pricing','docs','blog','dashboard','settings','checkout'][1+cityHash64(number,toUInt64(23))%8]),
        if(cityHash64(number,toUInt64(24))%4=0,NULL,toUInt32(100+cityHash64(number,toUInt64(25))%9900)),
        if(cityHash64(number,toUInt64(26))%3=0,
           map('campaign',['summer_sale','black_friday','spring_promo','launch'][1+cityHash64(number,toUInt64(27))%4],
               'source',['google','facebook','twitter','email','direct'][1+cityHash64(number,toUInt64(28))%5]),
           map('source',['google','facebook','twitter','email','direct'][1+cityHash64(number,toUInt64(29))%5])),
        if(cityHash64(number,toUInt64(30))%3=0,
           arrayFilter(x->x!='',[
             if(cityHash64(number,toUInt64(31))%4=0,'promo',''),
             if(cityHash64(number,toUInt64(32))%5=0,'vip_member',''),
             if(cityHash64(number,toUInt64(33))%6=0,'premium_user',''),
             if(cityHash64(number,toUInt64(34))%3=0,'mobile',''),
             if(cityHash64(number,toUInt64(35))%7=0,'new','')]),[]),
        ['US','DE','GB','FR','JP','BR','IN','CA','AU','NL'][1+cityHash64(number,toUInt64(36))%10],
        ['desktop','mobile','tablet'][1+cityHash64(number,toUInt64(37))%3],
        if(cityHash64(number,toUInt64(38))%5=0,toInt64(100+cityHash64(number,toUInt64(39))%9900),NULL)
        FROM numbers(1000000)"""
    (BASE / "fixture.sql").write_text(users_sql + ";\n\n" + events_sql + ";\n")
    query(users_sql, DATABASE)
    query(events_sql, DATABASE)
    query("OPTIMIZE TABLE events FINAL", DATABASE)
    query("OPTIMIZE TABLE users FINAL", DATABASE)
    query("SYSTEM STOP MERGES events", DATABASE)
    query("SYSTEM STOP MERGES users", DATABASE)
    query("CREATE USER IF NOT EXISTS budget_reader IDENTIFIED WITH no_password HOST LOCAL SETTINGS readonly=1, max_threads=2, max_memory_usage=2000000000, max_execution_time=10, max_result_rows=100000, max_result_bytes=16777216, result_overflow_mode='throw', output_format_json_quote_64bit_integers=0, output_format_json_quote_64bit_floats=0, date_time_output_format='iso', use_query_cache=0")
    query("GRANT SELECT ON exp04_budget.* TO budget_reader")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--create-data", action="store_true")
    args = parser.parse_args()
    if args.source:
        questions = build_questions(args.source)
        (BASE / "questions.json").write_text(json.dumps(questions, indent=2) + "\n")
    if args.create_data:
        create_data()
        (BASE / "build-options.tsv").write_text(query("SELECT * FROM system.build_options FORMAT TSVWithNames"))
        (BASE / "database-metadata.json").write_text(query("SELECT version() AS version, timezone() AS timezone, (SELECT count() FROM exp04_budget.events) AS events, (SELECT count() FROM exp04_budget.users) AS users FORMAT JSONEachRow"))
        fingerprints = query("SELECT 'events' AS tbl, count(), sum(cityHash64(*)) AS checksum FROM exp04_budget.events UNION ALL SELECT 'users', count(), sum(cityHash64(*)) FROM exp04_budget.users FORMAT JSONEachRow")
        (BASE / "data-fingerprints.jsonl").write_text(fingerprints)
        print("Created deterministic fixture: 1M events, 10K users (100 without events).", flush=True)


if __name__ == "__main__":
    main()
