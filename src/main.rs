use anyhow::{bail, Context, Result};
use regex::Regex;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use std::{
    env, fs, io,
    io::Write,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

#[derive(Debug, Clone)]
enum Backend {
    EdgeTts {
        voice: String,
    },
    OpenAiTts {
        api_key: String,
        model: String,
        voice: String,
        dry_run: bool,
    },
}

#[derive(Debug, Clone)]
struct AppConfig {
    pdf_path: PathBuf,
    backend: Backend,
    chunk_size: usize,
}

const VOICE: &str = "en-US-GuyNeural";
const DEFAULT_MAX_CHARS: usize = 12_000;
const OPENAI_TTS_1_RATE: f64 = 0.015; // per 1k chars
const OPENAI_TTS_1_HD_RATE: f64 = 0.03; // per 1k chars

fn clean_text(text: &str) -> String {
    let mut text = text.replace("\r\n", "\n").replace('\r', "\n");

    let hyphen_break = Regex::new(r"-\s*\n").unwrap();
    text = hyphen_break.replace_all(&text, "").into_owned();

    let multi_newline = Regex::new(r"\n{2,}").unwrap();
    text = multi_newline.replace_all(&text, "\n\n").into_owned();

    text = text.replace('\n', " ");

    let re_space = Regex::new(r"[ \t]+").unwrap();
    text = re_space.replace_all(&text, " ").into_owned();

    text.trim().to_string()
}

fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    // Split on sentence boundaries (., !, ?) followed by whitespace, without lookbehind.
    let mut sentences = Vec::new();
    let mut buf = String::new();
    let mut just_saw_end = false;

    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            just_saw_end = true;
        } else if just_saw_end && ch.is_whitespace() {
            let sentence = buf.trim();
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            buf.clear();
            just_saw_end = false;
        } else {
            just_saw_end = false;
        }
    }

    if !buf.trim().is_empty() {
        sentences.push(buf.trim().to_string());
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        if !current.is_empty() && current.len() + sentence.len() + 1 > max_chars {
            chunks.push(current.trim().to_string());
            current.clear();
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&sentence);
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}

fn extract_text_with_fallback(pdf_path: &PathBuf) -> Result<String> {
    let pdf_extract_err =
        match panic::catch_unwind(AssertUnwindSafe(|| pdf_extract::extract_text(pdf_path))) {
            Ok(Ok(text)) => return Ok(text),
            Ok(Err(err)) => {
                eprintln!("pdf-extract failed: {err}. Falling back to pdftotext...");
                Some(err.to_string())
            }
            Err(_) => {
                eprintln!("pdf-extract panicked. Falling back to pdftotext...");
                Some("pdf-extract panicked".to_string())
            }
        };

    let output = Command::new("pdftotext")
        .arg("-layout")
        .arg("-enc")
        .arg("UTF-8")
        .arg(pdf_path)
        .arg("-")
        .output()
        .context("Failed to run pdftotext. Is poppler-utils installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if let Some(err) = pdf_extract_err {
            bail!(
                "pdftotext failed (pdf-extract error: {err}): {}",
                stderr.trim()
            );
        }
        bail!("pdftotext failed: {}", stderr.trim());
    }

    let text = String::from_utf8(output.stdout).context("pdftotext output was not valid UTF-8")?;

    Ok(text)
}

fn prompt_with_default(prompt: &str, default: &str) -> Result<String> {
    print!("{} [{}]: ", prompt, default);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn prompt_yes_no(prompt: &str, default: bool) -> Result<bool> {
    let default_str = if default { "Y/n" } else { "y/N" };
    print!("{} [{}]: ", prompt, default_str);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let trimmed = input.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return Ok(default);
    }
    Ok(matches!(trimmed.as_str(), "y" | "yes"))
}

fn wizard(initial_path: Option<PathBuf>) -> Result<AppConfig> {
    let pdf_path = match initial_path {
        Some(p) => p,
        None => {
            let input = prompt_with_default("Path to PDF", "input.pdf")?;
            PathBuf::from(input)
        }
    };

    let chunk_input = prompt_with_default(
        "Chunk size (characters) – larger reduces pauses; too large may time out",
        &DEFAULT_MAX_CHARS.to_string(),
    )?;
    let chunk_size = chunk_input
        .trim()
        .parse::<usize>()
        .unwrap_or(DEFAULT_MAX_CHARS);

    let use_openai = prompt_yes_no("Use OpenAI TTS instead of edge-tts?", false)?;

    if use_openai {
        let env_key = env::var("OPENAI_API_KEY").ok();
        let api_key = match env_key {
            Some(k) if !k.trim().is_empty() && k.trim().starts_with("sk-") => k.trim().to_string(),
            _ => {
                print!("Enter OPENAI_API_KEY (starts with sk-): ");
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let key = input.trim();
                if key.is_empty() {
                    bail!("OPENAI_API_KEY is required for OpenAI TTS");
                }
                key.to_string()
            }
        };

        let model = prompt_with_default("OpenAI model (tts-1 or tts-1-hd)", "tts-1")?;
        let voice = prompt_with_default("OpenAI voice (e.g., alloy)", "alloy")?;
        let dry_run = prompt_yes_no("Dry run (estimate cost, no audio)?", false)?;

        Ok(AppConfig {
            pdf_path,
            backend: Backend::OpenAiTts {
                api_key,
                model,
                voice,
                dry_run,
            },
            chunk_size,
        })
    } else {
        Ok(AppConfig {
            pdf_path,
            backend: Backend::EdgeTts {
                voice: VOICE.to_string(),
            },
            chunk_size,
        })
    }
}

fn estimated_cost(chars: usize, model: &str) -> Option<f64> {
    let rate = match model {
        "tts-1" => Some(OPENAI_TTS_1_RATE),
        "tts-1-hd" => Some(OPENAI_TTS_1_HD_RATE),
        _ => None,
    }?;
    let units = (chars as f64) / 1000.0;
    Some(units * rate)
}

fn ensure_command_available(cmd: &str, args: &[&str], install_hint: &str) -> Result<()> {
    let status = Command::new(cmd)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    match status {
        Ok(_) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            bail!("`{cmd}` not found. Install it ({install_hint}).");
        }
        Err(err) => Err(err).with_context(|| format!("Failed to run `{cmd}`")),
    }
}

async fn synthesize_openai(
    client: &reqwest::Client,
    text: &str,
    model: &str,
    voice: &str,
    api_key: &str,
    output: &str,
    idx: usize,
) -> Result<()> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {}", api_key).parse().unwrap(),
    );

    let body = serde_json::json!({
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3"
    });

    let mut attempt = 0;
    let max_attempts = 4;
    loop {
        let resp = client
            .post("https://api.openai.com/v1/audio/speech")
            .headers(headers.clone())
            .json(&body)
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let bytes = r.bytes().await?;
                fs::write(output, &bytes)?;
                return Ok(());
            }
            Ok(r) => {
                attempt += 1;
                if attempt >= max_attempts {
                    let status = r.status();
                    let text = r.text().await.unwrap_or_default();
                    bail!(
                        "OpenAI TTS failed on chunk {}: status {} body {}. Check API key/model/voice and connectivity.",
                        idx,
                        status,
                        text
                    );
                }
                let backoff = Duration::from_secs(2u64.pow(attempt));
                println!(
                    "OpenAI TTS failed on chunk {} (status {}), retrying in {}s (attempt {}/{})",
                    idx,
                    r.status(),
                    backoff.as_secs(),
                    attempt,
                    max_attempts
                );
                thread::sleep(backoff);
            }
            Err(err) => {
                attempt += 1;
                if attempt >= max_attempts {
                    bail!("OpenAI TTS request failed on chunk {}: {}", idx, err);
                }
                let backoff = Duration::from_secs(2u64.pow(attempt));
                println!(
                    "OpenAI TTS error on chunk {}: {}. Retrying in {}s (attempt {}/{})",
                    idx,
                    err,
                    backoff.as_secs(),
                    attempt,
                    max_attempts
                );
                thread::sleep(backoff);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let initial_path = env::args().nth(1).map(PathBuf::from);
    let config = wizard(initial_path)?;

    ensure_command_available(
        "ffmpeg",
        &["-version"],
        "install ffmpeg from your package manager",
    )?;

    if let Backend::EdgeTts { .. } = config.backend {
        ensure_command_available("edge-tts", &["--help"], "pip install edge-tts")?;
    }

    let raw_text = extract_text_with_fallback(&config.pdf_path)
        .with_context(|| format!("Could not extract text from {}", config.pdf_path.display()))?;

    let text = clean_text(&raw_text);
    let chunks = chunk_text(&text, config.chunk_size);

    if let Backend::OpenAiTts {
        model,
        dry_run: true,
        ..
    } = &config.backend
    {
        let total_chars = text.chars().count();
        if let Some(cost) = estimated_cost(total_chars, model) {
            println!(
                "Dry run: ~{} chars, model {}, estimated cost ${:.2}",
                total_chars, model, cost
            );
        } else {
            println!(
                "Dry run: ~{} chars, model {} (no rate info)",
                total_chars, model
            );
        }
        return Ok(());
    }

    fs::create_dir_all("audio_chunks")?;

    println!("Found {} chunks.", chunks.len());

    let client = if let Backend::OpenAiTts { .. } = &config.backend {
        Some(reqwest::Client::builder().build()?)
    } else {
        None
    };

    for (i, chunk) in chunks.iter().enumerate() {
        let txt_file = format!("audio_chunks/chunk_{:04}.txt", i);
        let mp3_file = format!("audio_chunks/chunk_{:04}.mp3", i);

        if Path::new(&mp3_file).exists() {
            println!("Skipping {}, already exists", mp3_file);
            continue;
        }

        fs::write(&txt_file, chunk)?;

        println!("Creating {}", mp3_file);
        match &config.backend {
            Backend::EdgeTts { voice } => {
                let mut attempt = 0;
                let max_attempts = 4;
                loop {
                    let status = Command::new("edge-tts")
                        .args([
                            "--voice",
                            voice,
                            "--file",
                            &txt_file,
                            "--write-media",
                            &mp3_file,
                        ])
                        .status()
                        .context("Failed to run edge-tts. Is it installed?")?;

                    if status.success() {
                        break;
                    }

                    attempt += 1;
                    if attempt >= max_attempts {
                        anyhow::bail!("edge-tts failed on chunk {} after {} attempts", i, attempt);
                    }

                    let backoff = Duration::from_secs(2u64.pow(attempt));
                    println!(
                        "edge-tts failed on chunk {}, retrying in {}s (attempt {}/{})",
                        i,
                        backoff.as_secs(),
                        attempt,
                        max_attempts
                    );
                    thread::sleep(backoff);
                }
            }
            Backend::OpenAiTts {
                api_key,
                model,
                voice,
                ..
            } => {
                let client = client.as_ref().expect("client exists");
                synthesize_openai(client, chunk, model, voice, api_key, &mp3_file, i).await?;
            }
        }
    }

    let list_file = "audio_chunks/files.txt";

    let mut list = String::new();
    for i in 0..chunks.len() {
        list.push_str(&format!("file 'chunk_{:04}.mp3'\n", i));
    }

    fs::write(list_file, list)?;

    println!("Combining MP3 files...");

    let status = Command::new("ffmpeg")
        .current_dir("audio_chunks")
        .args([
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "files.txt",
            "-c",
            "copy",
            "../audiobook.mp3",
        ])
        .status()
        .context("Failed to run ffmpeg. Is it installed?")?;

    if !status.success() {
        anyhow::bail!("ffmpeg failed");
    }

    println!("Done: audiobook.mp3");

    Ok(())
}
