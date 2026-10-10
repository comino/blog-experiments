INSERT INTO users SELECT
        number + 1,
        toDateTime64('2023-01-01 00:00:00', 3, 'UTC') + toIntervalSecond(cityHash64(number, toUInt64(1)) % (730 * 86400)),
        ['free','pro','enterprise'][1 + cityHash64(number, toUInt64(2)) % 3],
        concat('user',toString(number+1),'@example.com'),
        if(cityHash64(number,toUInt64(3)) % 5 = 0, NULL, toUInt8(18+cityHash64(number,toUInt64(4))%50)),
        ['US','DE','GB','FR','JP','BR','IN','CA','AU','NL'][1+cityHash64(number,toUInt64(5))%10],
        map('signup_source',['organic','paid','referral','social'][1+cityHash64(number,toUInt64(6))%4])
        FROM numbers(10000);

INSERT INTO events SELECT
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
        FROM numbers(1000000);
