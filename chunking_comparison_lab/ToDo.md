# How to improve

# Tasks

## Phase 1: More PDFs Tests
Only the covid 6 page has golden dataset
- [ ] Generate golden dataset for all the PDFs
- [ ] Add logs to understand the "Why" behind test failures


## Phase 2: Structure Aware Strategy Table problem
1. Currently Full table is being stored in a single chunk in `strategies/structure_aware_chunking.py`.
2. If the chunk becomes too big, the embedding will ignore the table completely, or only fit the firest few row that are within the embedding size ignoring the latter rows.

- [ ] Split the table into smaller chunks while giving the table metadata like column names and table name to each chunk

**OR**
- [ ] Give the whole table to a LLM model and ask it to summarise the table and embed that summary, while keeping the table itself in the payload

## Phase 3: Polishing
- [ ] Create a web interface to run these tests and compare the results visually
- [ ] Pick the top stategy and let custom PDFs to be uploaded through the interface and test them