import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { HttpModule } from '@nestjs/axios';
import { AppController } from './app.controller';
import { ModelService } from './model.service';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    HttpModule.register({
      timeout: 120000,
      maxRedirects: 3,
    }),
  ],
  controllers: [AppController],
  providers: [ModelService],
})
export class AppModule {}
