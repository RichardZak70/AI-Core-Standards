#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import yaml from 'js-yaml';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

/**
 * Files to validate.
 * Add new entries here as your core standards grow.
 */
const files = [
  {
    label: 'models',
    schema: 'schemas/models.schema.json',
    data: 'templates/config/models.yaml'
  },
  {
    label: 'prompts',
    schema: 'schemas/prompts.schema.json',
    data: 'templates/config/prompts.yaml'
  }
];

/**
 * Create a single Ajv instance for all validations.
 * strict: 'log' will warn on schema issues without hard failing.
 * Switch to strict: true once schemas are stable.
 */
const ajv = new Ajv2020({
  strict: 'log',
  allErrors: true,
  allowUnionTypes: true
});
addFormats(ajv);

async function validateConfig({ label, schema, data }) {
  const schemaPath = path.join(repoRoot, schema);
  const dataPath = path.join(repoRoot, data);

  let schemaJson;
  try {
    const schemaContent = await readFile(schemaPath, 'utf-8');
    schemaJson = JSON.parse(schemaContent);
  } catch (err) {
    console.error(`❌ Failed to load/parse schema for '${label}': ${schemaPath}`);
    console.error(`   ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }

  const validate = ajv.compile(schemaJson);

  let parsedData;
  try {
    const dataContent = await readFile(dataPath, 'utf-8');
    parsedData = yaml.load(dataContent) ?? {};
  } catch (err) {
    console.error(`❌ Failed to load/parse data for '${label}': ${dataPath}`);
    console.error(`   ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }

  if (typeof parsedData !== 'object' || parsedData === null || Array.isArray(parsedData)) {
    console.error(`❌ ${label} config root must be an object: ${dataPath}`);
    return false;
  }

  const valid = validate(parsedData);

  if (!valid) {
    console.error(`❌ ${label} config failed schema validation: ${dataPath}`);
    for (const err of validate.errors ?? []) {
      const location = err.instancePath || '<root>';
      const message = err.message ?? 'validation error';
      console.error(`  - ${location}: ${message}`);
    }
    return false;
  }

  console.log(`✅ ${label} config matches ${schema}`);
  return true;
}

async function main() {
  let hasErrors = false;

  for (const fileDef of files) {
    const ok = await validateConfig(fileDef);
    if (!ok) {
      hasErrors = true;
    }
  }

  if (hasErrors) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('💥 Unexpected error during schema validation:');
  console.error(err);
  process.exit(1);
});
