// q_format_config.h
#ifndef Q_FORMAT_CONFIG_H
#define Q_FORMAT_CONFIG_H

#ifndef Q_SHIFT
#define Q_SHIFT 9    // 預設值，如果外部沒定義
#endif


#ifndef Q_SCALE
#define Q_SCALE (1 << (Q_SHIFT - 1))    // 預設值，如果外部沒定義
#endif

#endif
