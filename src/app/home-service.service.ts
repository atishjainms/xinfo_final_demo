import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
@Injectable({
  providedIn: 'root'
})
export class HomeServiceService {

  constructor(
  private http: HttpClient) {

  }
  newinformation='assets/getNewsInformation.json';

getNewsInformation(){

    return this.http.get<any[]>(this.newinformation);

}





}
