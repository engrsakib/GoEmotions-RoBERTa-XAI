import { HttpService } from '@nestjs/axios';
import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AxiosError } from 'axios';
import { firstValueFrom } from 'rxjs';
import {
  ChatResponse,
  ExplainResponse,
  HealthResponse,
  PredictResponse,
} from './dto/text.dto';

@Injectable()
export class ModelService {
  private readonly logger = new Logger(ModelService.name);
  private readonly modelBaseUrl: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
  ) {
    this.modelBaseUrl =
      this.configService.get<string>('MODEL_SERVICE_URL') ?? 'http://model:8000';
  }

  async health(): Promise<HealthResponse> {
    try {
      const response = await firstValueFrom(
        this.httpService.get(`${this.modelBaseUrl}/health`),
      );
      return {
        status: 'ok',
        gateway: 'up',
        model: response.data,
      };
    } catch (error) {
      this.logger.error('Model health check failed', error);
      throw new ServiceUnavailableException('Model service unavailable');
    }
  }

  async predict(text: string): Promise<PredictResponse> {
    return this.post<PredictResponse>('/predict', { text });
  }

  async explain(text: string): Promise<ExplainResponse> {
    return this.post<ExplainResponse>('/explain', { text });
  }

  async chat(text: string): Promise<ChatResponse> {
    return this.post<ChatResponse>('/chat', { text });
  }

  private async post<T>(path: string, payload: { text: string }): Promise<T> {
    try {
      const response = await firstValueFrom(
        this.httpService.post<T>(`${this.modelBaseUrl}${path}`, payload),
      );
      return response.data;
    } catch (error) {
      const axiosError = error as AxiosError<{ detail?: string }>;
      const detail =
        axiosError.response?.data?.detail ??
        axiosError.message ??
        'Model service request failed';
      this.logger.error(`Model ${path} failed: ${detail}`);
      throw new ServiceUnavailableException(detail);
    }
  }
}
