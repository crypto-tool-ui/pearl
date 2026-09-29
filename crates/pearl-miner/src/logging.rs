//! Plain Japanese console output; protocol errors retain their original diagnostic detail.
use std::fmt;
use tracing::{
    field::{Field, Visit},
    Event, Subscriber,
};
use tracing_subscriber::{
    fmt::{
        format::{FormatEvent, FormatFields, Writer},
        time::{FormatTime, SystemTime},
        FmtContext,
    },
    registry::LookupSpan,
};

pub struct JapaneseText;
impl<S, N> FormatEvent<S, N> for JapaneseText
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        SystemTime.format_time(&mut writer)?;
        let level = match *event.metadata().level() {
            tracing::Level::ERROR => "エラー",
            tracing::Level::WARN => "警告",
            tracing::Level::INFO => "情報",
            tracing::Level::DEBUG => "デバッグ",
            tracing::Level::TRACE => "トレース",
        };
        write!(writer, " [{level}] ")?;
        let mut fields = JapaneseFields {
            writer: &mut writer,
            result: Ok(()),
        };
        event.record(&mut fields);
        fields.result?;
        writeln!(writer)
    }
}
struct JapaneseFields<'a, 'w> {
    writer: &'a mut Writer<'w>,
    result: fmt::Result,
}
impl JapaneseFields<'_, '_> {
    fn write(&mut self, field: &Field, value: impl fmt::Display) {
        if self.result.is_err() {
            return;
        }
        self.result = if field.name() == "message" {
            write!(self.writer, "{value}")
        } else {
            let label = match field.name() {
                "device" => "GPU番号",
                "job_id" => "ジョブ",
                "scanned" => "処理済み",
                "error" => "エラー詳細",
                "code" => "終了コード",
                "addr" => "待受アドレス",
                "kind" => "拒否種別",
                "reason" => "理由",
                "t_rows" => "行位置",
                "t_cols" => "列位置",
                "id" => "要求ID",
                "accepted" => "受理",
                name => name,
            };
            write!(self.writer, " | {label}={value}")
        };
    }
}
impl Visit for JapaneseFields<'_, '_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.write(field, format_args!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.write(field, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io,
        sync::{Arc, Mutex},
    };
    #[derive(Clone)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl io::Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn text_is_japanese_and_logging_can_be_disabled() {
        for enabled in [false, true] {
            let bytes = Arc::new(Mutex::new(Vec::new()));
            let output = bytes.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_ansi(false)
                .event_format(JapaneseText)
                .with_max_level(if enabled {
                    tracing::level_filters::LevelFilter::INFO
                } else {
                    tracing::level_filters::LevelFilter::OFF
                })
                .with_writer(move || Capture(output.clone()))
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(device = 0, "採掘状況 | ハッシュレート=934.507 MH/s")
            });
            let text = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
            if enabled {
                assert!(
                    text.contains("[情報] 採掘状況 | ハッシュレート=934.507 MH/s | GPU番号=0"),
                    "{text}"
                );
                assert!(!text.contains("\"timestamp\""));
            } else {
                assert!(text.is_empty());
            }
        }
    }
}
