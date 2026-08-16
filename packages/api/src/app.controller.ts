import { Body, Controller, Get, Post } from '@nestjs/common';
import { TextDto } from './dto/text.dto';
import { ModelService } from './model.service';

@Controller()
export class AppController {
  constructor(private readonly modelService: ModelService) {}

  @Get('health')
  health() {
    return this.modelService.health();
  }

  @Post('predict')
  predict(@Body() payload: TextDto) {
    return this.modelService.predict(payload.text);
  }

  @Post('explain')
  explain(@Body() payload: TextDto) {
    return this.modelService.explain(payload.text);
  }

  @Post('chat')
  chat(@Body() payload: TextDto) {
    return this.modelService.chat(payload.text);
  }

  @Post('grok/classify')
  classify(@Body() payload: TextDto) {
    return this.modelService.explain(payload.text);
  }
}
